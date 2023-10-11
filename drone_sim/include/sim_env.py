import matplotlib.pyplot as plt
import numpy as np
import os
import csv
import json  
import random 
import torch 
import cvxpy as cp
import pypose as pp
from tqdm import tqdm
from scipy.stats import multivariate_normal
from include.controllers import (
    get_k_dtcbf_account_for_jensens, 
    get_k_dtcbf_pendulum,
    get_k_dtcbf_variable_jensens
)

from include.cbfs import (
    setup_drone_floor_barrier
)

from include.systems import (
        system_types, 
        LinearCBF
)

from include.probability_utils import (
    get_kushner_bounds,
    get_learned_mean_cov_generator, 
    get_learned_mean_cov_generator_mlp,     
)

def keys_to_list(dict_keys):
    keys_list = [] 
    for k in dict_keys: 
        keys_list.append(k)
    return keys_list

class SimEnv: 

    """ 
        Sim Environment Class 

        Attributes: 
            time_step: time interval for integration 
            
    """

    def __init__(self, params):
        print("Initializing Sim Environment")
        print("\tSystem Type: " + params["system type"])
        print("\tTime Step: " + str(params["time step"]))
        print("\tIntegration Scheme: " + params["integration scheme"])
        print("\tDisturbance Estimator: " + params["safety params"]["disturbance estimator"])

        self.time_step = params["time step"]
        self.system_type = params["system type"]
        self.integration_scheme = params["integration scheme"]
        self.sim_time_span = params["simulation time span"]
        self.sim_time_steps = int(self.sim_time_span/self.time_step)
        self.safety_params = params["safety params"]
        self.cut_short_if_unsafe = params["cut short if unsafe"]

        self.setup_noise(params)
        self.setup_system()
        self.setup_safety(params)


    def setup_system(self):        
        system_type = self.system_type
        if system_type in system_types.keys():  
            self.n_state_size = system_types[system_type]["n state size"]
            if "p disturbance size" in system_types[system_type].keys():
                self.p_disturbance_size = system_types[system_type]["p disturbance size"]
            else: 
                self.p_disturbance_size = self.n_state_size
            self.m_input_size = system_types[system_type]["m input size"]
            if self.integration_scheme == "discrete": 
                self.F = system_types[system_type]["get F"](self.time_step, self.process_noise)
            else:
                self.f = system_types[system_type]["f"]
                self.g = system_types[system_type]["g"]
            self.x0 = system_types[system_type]["initial condition"]
            self.k = system_types[system_type]["k default"]
            self.k_nom = self.k
            self.labels =  {
                "state"  : system_types[system_type]["state labels"], 
                "input"  : system_types[system_type]["input labels"], 
                "slack"  : system_types[system_type]["slack labels"],
                "safety" : "h(x)"
            }
        
        if system_type == "drone": 
            self.drone_add_disturbance = system_types[system_type]["add disturbance"]    
            self.F_eul = system_types[system_type]["get F eul"](self.time_step)
        self.sim_iteration = -1
        self.sim_results = []


    def setup_noise(self, params): 
        # Set Random Seed
        random.seed(params["random seed"])
        np.random.seed(params["random seed"])
        torch.manual_seed(params["random seed"])

        self.n_state_size = system_types[self.system_type]["n state size"]

        self.process_noise = dict()
        self.process_noise["on/off"] = params["process noise params"]["on/off"]
        self.process_noise["deterministic on/off"] = params["process noise params"]["deterministic on/off"]

        if params["process noise params"]["deterministic on/off"]:
            self.samples = params["process noise params"]["deterministic disturbance"]*np.ones(((self.sim_time_steps+1) * params["number of trials"] ,1))
        else:
            self.process_noise["mean"] = params["process noise params"]["mean"]
            self.process_noise["covariance"] = params["process noise params"]["covariance"]
            if self.system_type == "drone": 
                z_distribution = multivariate_normal(mean = np.zeros(9), cov = np.eye(9))
            else: 
                z_distribution = multivariate_normal(mean = np.zeros(self.n_state_size), cov = np.eye(self.n_state_size))

            self.samples = z_distribution.rvs((self.sim_time_steps+1) * params["number of trials"] )
        
        def sample_distribution(x): 
            # sample = np.array([distribution.rvs(1)]).T
            sample = self.samples[-1]
            cov = self.process_noise["covariance"](x)
            L = np.linalg.cholesky(cov)
            sample = np.array([self.process_noise["mean"](x)]).T + L @ np.expand_dims(sample, axis=0).T
            self.samples = self.samples[:-1]
            return sample
        
        self.process_noise["get sample"] = sample_distribution
    
    def setup_safety(self, params): 

        # Define CBF 
        if params["safety params"]["cbf"] == "one ball": 
            self.h = lambda x : 1 - cp.norm(x)**2
        elif params["safety params"]["cbf"] == "lyapunov pendulum": 
            # k to keep theta in [-pi/6, pi/6]
            k = 6**2/np.pi**2/np.sqrt(3)
            P = np.array([[np.sqrt(3), 1],[1, np.sqrt(3)]])
            self.h = lambda x: 1 - k*cp.quad_form(x, P)
        elif params["safety params"]["cbf"] == "drone floor": 
            self.h, self.dhdx_max_eig = setup_drone_floor_barrier() 


        if self.system_type == "drone":
            if params["safety params"]["controller type"] == "dtcbf":
                # Get disturbance mean and tr(cov) estimators 
                if params["safety params"]["disturbance estimator"] == "learned":
                    if params["safety params"]["model type"] == "cvae":
                        mean_cov_generator = get_learned_mean_cov_generator(params["safety params"]["alpha"], self.sim_time_span)
                    else: 
                        mean_cov_generator = get_learned_mean_cov_generator_mlp(params["safety params"]["alpha"], self.sim_time_span)
                elif params["safety params"]["disturbance estimator"] == "zero":
                    def mean_cov_generator(state): 
                        mean = torch.zeros((self.p_disturbance_size,1), dtype=torch.float64)
                        trace_cov = torch.tensor([[0]], dtype=torch.float64)
                        return mean, trace_cov
                elif params["safety params"]["disturbance estimator"] == "constant":
                    def mean_cov_generator(state): 
                        mean =  torch.tensor(np.load("./results/drone_discrete_dT_" + str(params["time step"]) + "_N_" + str(params["simulation time span"]) + "_alpha_" + str(params["safety params"]["alpha"]) + "_system_drone_disturbanceEstimator_zero_controller_SE3_modeltype_none_Experiment_0/d_mean.npy")).reshape(self.p_disturbance_size,1)
                        # mean = mean.float()
                        trace_cov = torch.tensor(np.load("results/drone_discrete_dT_" + str(params["time step"]) + "_N_" + str(params["simulation time span"]) + "_alpha_" + str(params["safety params"]["alpha"]) + "_system_drone_disturbanceEstimator_zero_controller_SE3_modeltype_none_Experiment_0/d_cov.npy")).trace().reshape(1,1)
                        # trace_cov = trace_cov.float() 
                        return mean, trace_cov
                        
                elif params["safety params"]["disturbance estimator"] == "true":
                    def mean_cov_generator(state): 
                        mean = torch.tensor(self.process_noise["mean"](state), dtype=torch.float64).reshape(self.p_disturbance_size,1)
                        # mean = mean.float()
                        trace_cov = torch.tensor(self.process_noise["covariance"](state), dtype=torch.float64).trace().reshape(1,1)
                        # trace_cov = trace_cov.float() 
                        return mean, trace_cov

                self.k = get_k_dtcbf_variable_jensens(params, self.n_state_size, self.m_input_size, self.h, self.k_nom, self.F_eul, mean_cov_generator, self.dhdx_max_eig)
        
            elif params["safety params"]["controller type"] == "SE3":
                self.k = system_types[self.system_type]["SE(3) controller"]

        else: 
            print("Sorry, this sim only works for the drone at the moment. Please choose system type: drone")            

        
        self.kushner_bound = get_kushner_bounds(N=self.sim_time_steps, 
                                                alpha=params["safety params"]["alpha"],
                                                d_min=0, 
                                                M=params["safety params"]["cbf max"], 
                                                h_0=self.h(self.x0).value)


    def dxdt_dynamics(self, x): 
        """
            Calculate the Control Affine Dynamics
        """
        dxdt = self.f(x) + self.g(x) @  self.k(x) + self.process_noise["get sample"]()*self.process_noise["on/off"]

        return dxdt 


    def step(self, x_now): 
        if self.integration_scheme == "Euler": 
            x_next = self.euler_step(x_now)
        elif self.integration_scheme == "discrete": 
            u, slack = self.k(x_now)
            x_next = self.F(x_now, u)
            if self.process_noise["deterministic on/off"]:
                x_next += self.process_noise["get sample"]()*self.process_noise["deterministic on/off"]   
            else:
                if self.system_type == "drone": 
                    x_next = self.drone_add_disturbance(x_next, self.process_noise["get sample"](x_now)*self.process_noise["on/off"])
                else: 
                    x_next += self.process_noise["get sample"]()*self.process_noise["on/off"] 
        else: 
            raise("Sorry, this integration scheme is not implemented yet.")

        return x_next, u, slack

    def euler_step(self, x_now): 
        dxdt = self.dxdt_dynamics(x_now)
        return x_now + dxdt*self.time_step
    
    def run_experiment(self, params): 
        # runs multiple simulations

        for i in tqdm(range(params["number of trials"])): 
            # print("Trial " + str(i+1) + " of " + str(params["number of trials"]) )
            self.simulate()

    def simulate(self): 
        # print("Beginning " +str(self.sim_time_span) + " second Simulation at " + str(1/self.time_step) + " Hz")
        try: self.h
        except: raise("Please define the safety function self.h before running the simulation")
        
        self.sim_iteration+=1 
        self.sim_results.append(dict())

        # Initialize Results
        x = self.x0.to("cuda")
        time = 0 
        self.sim_results[self.sim_iteration]["time"] = [time]
        self.sim_results[self.sim_iteration]["state trajectory"] = [x]
        u, slack = self.k(x)
        self.sim_results[self.sim_iteration]["input trajectory"] = [u]
        self.sim_results[self.sim_iteration]["slack trajectory"] = [slack]
        self.sim_results[self.sim_iteration]["safety trajectory"] = [self.h(x).value[0,0]]

        # Run Simulation
        for i in range(self.sim_time_steps): 
            x_old = x
            x,u,slack = self.step(x_old)
            time += self.time_step
            self.sim_results[self.sim_iteration]["time"].append(time)
            self.sim_results[self.sim_iteration]["state trajectory"].append(x_old)
            self.sim_results[self.sim_iteration]["input trajectory"].append(u)
            self.sim_results[self.sim_iteration]["slack trajectory"].append(slack)
            self.sim_results[self.sim_iteration]["safety trajectory"].append(self.h(x_old).value[0,0])
            if self.h(x).value[0,0] < 0 and self.cut_short_if_unsafe: 
                break

        self.format_results()


    def format_results(self): 
        self.sim_results[self.sim_iteration]["time"] = np.array([self.sim_results[self.sim_iteration]["time"]])
        self.sim_results[self.sim_iteration]["state trajectory"] = np.hstack(self.sim_results[self.sim_iteration]["state trajectory"])
        self.sim_results[self.sim_iteration]["input trajectory"] = np.hstack(self.sim_results[self.sim_iteration]["input trajectory"])
        self.sim_results[self.sim_iteration]["slack trajectory"] = np.hstack(self.sim_results[self.sim_iteration]["slack trajectory"])
        self.sim_results[self.sim_iteration]["safety trajectory"] = np.array([self.sim_results[self.sim_iteration]["safety trajectory"]])

    def set_results_folder(self): 
        if hasattr(self,"new_folder_name"): 
            new_folder_name = self.new_folder_name
        else: 
            # Create a new folder for the results 
            base_folder_name = "./results/" + self.system_type + "_" + self.integration_scheme
            base_folder_name += "_dT_" + str(self.time_step)
            base_folder_name += "_N_" + str(self.sim_time_span) 
            base_folder_name += "_alpha_" + str(self.safety_params["alpha"])
            base_folder_name += "_system_" + str(self.system_type)
            base_folder_name += "_disturbanceEstimator_" + str(self.safety_params["disturbance estimator"])
            base_folder_name += "_controller_" + str(self.safety_params["controller type"])
            base_folder_name += "_modeltype_" + str(self.safety_params["model type"])
            for i in range(1024): 
                if i == 1023: 
                    raise("Please clean up the results folder")
                new_folder_name = base_folder_name + "_Experiment_" + str(i)
                # if not os.path.isdir("./results/"+self.safety_params["controller type"]): 
                #     os.mkdir("./results/"+self.safety_params["controller type"])
                if not os.path.isdir(new_folder_name): 
                    os.mkdir(new_folder_name)
                    break                        
            self.new_folder_name = new_folder_name



        return new_folder_name



    def save_plots(self, params, verbose, dpi=100): 

        # Save Results
        new_folder_name = self.set_results_folder()

        # Plot States 
        if verbose:
            print("\tSaving State Plot")
        self.save_plot("state", self.n_state_size, new_folder_name, dpi)

        # Plot Inputs 
        if verbose:
            print("\tSaving Input Plot")
        self.save_plot("input", self.m_input_size, new_folder_name, dpi)

        # Plot Inputs 
        if verbose:
            print("\tSaving Slack Plot")
        self.save_plot("slack", 1, new_folder_name, dpi)

        # Plot Safety
        if verbose:
            print("\tSaving Safety Plot")
        self.save_plot("safety", 1, new_folder_name, dpi)

    # Save Plot to File
    #   Expects "type" to be ["state", "input", "safety"]
    def save_plot(self, type, size, new_folder_name, dpi = 100):
        dict_string = type + " trajectory"
        fig, axs = plt.subplots(size, 1, sharex="col")
        if size == 1: 
            axs = [axs] 
        
        for sim_idx in range(self.sim_iteration+1): 
            for i, ax in enumerate(axs): 
                try: 
                    ax.plot(self.sim_results[sim_idx]["time"][0,:], self.sim_results[sim_idx][dict_string][i,:], alpha=0.1, color="b", linewidth=1)
                except: 
                    breakpoint()
                ax.set_ylabel(self.labels[type][i])
        fig.suptitle( type + " trajectories" )
        axs[-1].set_xlabel("Time [sec]")

        # Plot 0 Line on CBF plot
        if type == "safety": 
            axs[-1].hlines(
                y=0, 
                xmin = self.sim_results[self.sim_iteration]["time"][0,0], 
                xmax = self.sim_time_span,
                colors="k", 
                linestyles="dashed"
            )
            axs[-1].set_ylim((-0.1, 1))

        plt.savefig(new_folder_name + "/" + type + ".png", dpi = dpi)
        plt.close()



    # Save Results to a CSV File 
    def save_results(self, params, verbose): 

        new_folder_name = self.set_results_folder()
        if verbose:
            print("Saving Results to " + new_folder_name)
        if verbose:
            print("\tSaving Experiment Parameters")
        self.save_params(params, new_folder_name)

        if verbose:
            print("\tSaving Results to CSV")
        file_name = new_folder_name + "/results.csv"
        w = csv.writer(open(file_name, "w"))
        keys = self.sim_results[self.sim_iteration].keys()

        row = ["iteration"]
        row.extend(["time"])
        row.extend(self.labels["state"])
        row.extend(self.labels["input"])
        row.extend("h")
        w.writerow(row)
        
        sizes = {
            "state trajectory" : self.n_state_size, 
            "input trajectory" : self.m_input_size, 
            "slack trajectory" : 1,
            "safety trajectory": 1, 
            "time" : 1
        }

        for sim_idx in range(self.sim_iteration+1):
            # Cute short plotting if sim didn't finish because it was unsafe
            for t_idx in range(self.sim_results[sim_idx]['time'].shape[1]): 
                row = [sim_idx]
                for key in keys: 
                    for n_idx in range(sizes[key]): 
                        row.append(self.sim_results[sim_idx][key][n_idx, t_idx])
                w.writerow(row)
                if self.sim_results[sim_idx]["safety trajectory"][0, t_idx]<0 and self.cut_short_if_unsafe:
                    break

    def save_params(self, params, new_folder_name):
        with open(new_folder_name+"/params.json", "w") as json_file: 
            json.dump(params, json_file, indent=4)


    # Save Metrics 

    def save_metrics(self, params, verbose): 
        new_folder_name = self.set_results_folder()

        metrics = self.get_metrics(self.sim_results)
        with open(new_folder_name + "/" + "metrics.json", "w") as json_file: 
            json.dump(metrics, json_file, indent=4)

    def combine_batch_results(self, params): 
        controller_results_folder_name = "./results/" + self.safety_params["controller type"] + "/"
    
        trials = os.listdir(controller_results_folder_name)

        batch_csv_file = csv.writer(open(controller_results_folder_name + "../"+self.safety_params["controller type"]+"_batched_results.csv", "w"))

        for trial_idx, trial_name in enumerate(trials): 
            csv_results_filename = controller_results_folder_name + trial_name + "/results.csv"
            csv_file = csv.reader(open(csv_results_filename, "r"))
            for i, row in enumerate(csv_file):
                if i == 0 and trial_idx == 0 : 
                    batched_row = ["trial number"]
                    batched_row.extend(row)
                    batch_csv_file.writerow(batched_row)
                elif i>0: 
                    batched_row = [trial_idx]
                    batched_row.extend(row)
                    batch_csv_file.writerow(batched_row)

        os.system("rm -rf "+ controller_results_folder_name)

    def analyze_batch(self, params):        
        batch_csv_file = "./results/" + params["safety params"]["controller type"]+"_batched_results.csv"
        batch_csv_file = csv.DictReader(open(batch_csv_file,"r"))

        trial_number = -1 

        results = dict()
        for row in batch_csv_file:
            if str(trial_number) != row["trial number"]: 
                trial_number+=1
                results[str(trial_number)] = {
                    "state trajectory" : [],
                    "input trajectory" : [],
                    "safety trajectory": [],
                    "time" : []
                }
            
            state = []
            for component in self.labels["state"]:
                state.append(row[component])
            
            input = []
            for component in self.labels["input"]: 
                input.append(row[component])

            results[str(trial_number)]["state trajectory"].append(state)
            results[str(trial_number)]["input trajectory"].append(input)
            results[str(trial_number)]["safety trajectory"].append(row["h"])
            results[str(trial_number)]["time"].append(row["time"])

        for key in results.keys(): 
            results[key]["state trajectory"] = np.array(results[key]["state trajectory"]).astype(np.float).T
            results[key]["input trajectory"] = np.array(results[key]["input trajectory"]).astype(np.float).T
            results[key]["safety trajectory"] = np.array([results[key]["safety trajectory"]]).astype(np.float)
            results[key]["time"] = np.array([results[key]["time"]]).astype(np.float)


        print("\tSaving Batch Plots")
        base_folder_name = "./results/" + params["safety params"]["controller type"]
        for i in range(1024): 
            if i == 1023: 
                raise("Please clean up the results folder")
            new_folder_name = base_folder_name + str(i)
            if not os.path.isdir(new_folder_name): 
                os.mkdir(new_folder_name)
                break

        os.system("mv " + "./results/"+params["safety params"]["controller type"] +"_batched_results.csv " + new_folder_name + "/batched_results.csv")
        self.save_batch_plot(results, "state", self.n_state_size, new_folder_name + "/")
        self.save_batch_plot(results, "input", self.m_input_size, new_folder_name + "/")
        self.save_batch_plot(results, "safety", 1, new_folder_name + "/")

        metrics = self.get_metrics(results)
        with open(new_folder_name + "/" + "metrics.json", "w") as json_file: 
            json.dump(metrics, json_file, indent=4)

        print("\tSaving Batch Parameters")
        self.save_params(params, new_folder_name)


        
    # Save Plot to File
    #   Expects "type" to be ["state", "input", "safety"]
    def save_batch_plot(self, results, type, size, file_prefix):
        dict_string = type + " trajectory"

        fig, axs = plt.subplots(size, 1, sharex="col")
        if size == 1: 
            axs = [axs] 
        
        for trial_number in results.keys(): 
            for i, ax in enumerate(axs): 
                ax.plot(
                    results[trial_number]["time"][0,:], 
                    results[trial_number][dict_string][i,:], 
                    color="blue", 
                    alpha=0.1, 
                    linewidth=2
                )
                ax.set_ylabel(self.labels[type][i])
            fig.suptitle( type + " trajectories" )
            axs[-1].set_xlabel("Time [sec]")

        # Plot 0 Line on CBF plot
        if type == "safety": 
            axs[-1].hlines(
                y=0, 
                xmin = results["0"]["time"][0,0], 
                xmax = results["0"]["time"][0,-1], 
                colors="k", 
                linestyles="dashed"
            )

        plt.savefig(file_prefix + "batch_" + type + ".png")
        plt.close()

    def get_metrics(self, results): 
        metrics = {
            "avg safety violations per trial" : 0, 
            "min cbf" : np.inf, 
            "max abs(input)" : 0, 
            "probability trial is unsafe" : 0, 
        }

        gammas = [0, 3, 8]
        for g_idx, gamma in enumerate(gammas): 
            metrics["gamma="+"%.0f"%gamma+" violations"] = 0
            metrics["gamma="+"%.0f"%gamma+" violations kushner"] = self.kushner_bound(gamma)


        for sim_trial in self.sim_results: 
            h_min_trial = sim_trial["safety trajectory"].min()
            u_max_trial = np.abs(sim_trial["input trajectory"]).max()
            count_safety_violations_trial = np.sum(sim_trial["safety trajectory"] < 0)

            if h_min_trial < metrics["min cbf"]: 
                metrics["min cbf"] = h_min_trial

            if u_max_trial > metrics["max abs(input)"]: 
                metrics["max abs(input)"] = u_max_trial

            metrics["avg safety violations per trial"] += count_safety_violations_trial

            if h_min_trial < 0: 
                metrics["probability trial is unsafe"] += 1

            for g_idx, gamma in enumerate(gammas): 
                if h_min_trial < -gamma: 
                    metrics["gamma="+"%.0f"%gamma+" violations"] += 1
                

        metrics["avg safety violations per trial"] /= (len(self.sim_results)*self.sim_time_steps)
        metrics["probability trial is unsafe"] /= len(self.sim_results)

        for g_idx, gamma in enumerate(gammas): 
            metrics["gamma="+"%.0f"%gamma+" violations"] /= len(self.sim_results)

        return metrics


