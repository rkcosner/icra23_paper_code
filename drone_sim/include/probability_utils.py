import torch 
import numpy as np
from sklearn.preprocessing import StandardScaler

import sys 
sys.path.insert(0, "../icra2023_disturbance_learning/")
from cvae_model import * 


def disturbance_mean(x): 
    return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


def disturbance_cov(x): 
    return np.eye(9) * (1e-5 +  np.exp(-30*x[2,0]**2) * 5e-4 ) 

latent_dim = 10 
obs_dim = 9 
input_dim = 10 
hidden_layers = [200, 200]

def get_learned_mean_cov_generator(alpha, T): 

    filename ="./results/drone_discrete_dT_0.003_N_" + str(T) + "_alpha_" + str(alpha) + "_system_drone_disturbanceEstimator_zero_controller_SE3_modeltype_none_Experiment_0"


    input_scaler = StandardScaler()
    input_scaler.mean_ = np.load(filename + "/input_mean.npy")
    input_scaler.var_ = np.load(filename + "/input_var.npy")
    input_scaler.scale_ = np.load(filename + "/input_scale.npy")

    target_scaler = StandardScaler()
    target_scaler.mean_ = np.load(filename    + "/target_mean.npy")
    target_scaler.var_ = np.load(filename     + "/target_var.npy")
    target_scaler.scale_ = np.load(filename   + "/target_scale.npy")

    model = CVAE(latent_dim, obs_dim, input_dim, hidden_layer_sizes=hidden_layers)
    model.load_state_dict(torch.load(filename+"/cvae_weights.pth"))

    def get_learned_mean_cov(state):
        # negative quaternion convention
        if state[3,0] > 0: 
            state[3:7,0] *=-1
        c_scaled = input_scaler.transform(state.T)
        c_scaled = torch.tensor(c_scaled, dtype=torch.float32)
        mu_gen, cov_gen = model.get_mean_cov_and_rescale(10000, c_scaled, target_scaler)

        return mu_gen, cov_gen

    return get_learned_mean_cov



def get_learned_mean_cov_generator_mlp(alpha, T): 

    filename ="./results/drone_discrete_dT_0.003_N_" + str(T) + "_alpha_" + str(alpha) + "_system_drone_disturbanceEstimator_zero_controller_SE3_modeltype_none_Experiment_0"


    input_scaler = StandardScaler()
    input_scaler.mean_ = np.load(filename + "/input_mean.npy")
    input_scaler.var_ = np.load(filename + "/input_var.npy")
    input_scaler.scale_ = np.load(filename + "/input_scale.npy")

    target_scaler = StandardScaler()
    target_scaler.mean_ = np.load(filename    + "/target_mean.npy")
    target_scaler.var_ = np.load(filename     + "/target_var.npy")
    target_scaler.scale_ = np.load(filename   + "/target_scale.npy")

    model = MLP(input_dim, obs_dim, hidden_layer_sizes=[100, 100], nonlinearity=nn.Softplus(beta=2))
    model.load_state_dict(torch.load(filename+"/mlp_weights.pth"))

    def get_learned_mean_cov(state):
        # negative quaternion convention
        if state[3,0] > 0: 
            state[3:7,0] *=-1

        c_scaled = input_scaler.transform(state.T)
        c_scaled = torch.tensor(c_scaled, dtype=torch.float32)
        mu_gen = model(c_scaled) * target_scaler.scale_ + target_scaler.mean_
        cov_gen = torch.eye(9) * 0 # MLP estimates no covariance

        mu_gen = mu_gen.T

        return mu_gen, cov_gen

    return get_learned_mean_cov



# Kushner Probability Bounds from the RSS Paper (https://arxiv.org/pdf/2302.07469.pdf)
def get_kushner_bounds(N, alpha, d_min, M, h_0):

    # def kushner_loose(gamma): 

    #     Psi = alpha**N

    #     C  = (M - h_0)/(M + gamma)*Psi
    #     C += (M - d_min/(1-alpha))/(M + gamma) * (1 - Psi) 

    #     return C  

    def kushner_bound(gamma): 
        if d_min >= -gamma*(1-alpha): 
            C = (h_0 + gamma)/(M + gamma)
            for i in range(N): 
                C *= (M*alpha + gamma + d_min)/(M + gamma)
            return (1 - C)[0,0].item() 
        else: 
            Psi = alpha**N

            C  = (M - h_0)/(M + gamma)*Psi
            C += (M - d_min/(1-alpha))/(M + gamma) * (1 - Psi) 

            return (C)[0,0].item()


    return kushner_bound