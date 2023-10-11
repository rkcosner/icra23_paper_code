import numpy as np 
import scipy 
import torch
import pypose as pp

from include.systems import (
        k_default_1D_double_integrator, 
        A_1D_double_integrator, 
        B_1D_double_integrator
    )

def get_k_discrete_time_euler_cbf_1D_double_integrator(params, cbf, k_nominal): 
    A_euler = A_1D_double_integrator * params["time step"]
    B_euler = B_1D_double_integrator * params["time step"]
    a = np.array(params["safety params"]["linear cbf a"])
    b = params["safety params"]["linear cbf b"]
    alpha = params["safety params"]["alpha"]


    def k_cbf(x_now):
        h =  cbf(x_now)
        u_nominal = k_nominal(x_now)
        if a @ (A_euler @ x_now + B_euler @ u_nominal + x_now) + b - alpha * h >= 0: 
            return u_nominal
        else: 
            u_star = B_euler.T @ a.T * (alpha * h - a @ (A_euler @ x_now + x_now)- b)
            u_star /= a @ B_euler @ B_euler.T @ a.T  
            return u_star


    return k_cbf


def get_k_discrete_time_euler_state_constraint_1D_double_integrator(params, cbf, k_nominal): 
    A_euler = A_1D_double_integrator * params["time step"]
    B_euler = B_1D_double_integrator * params["time step"]
    a = np.array(params["safety params"]["linear cbf a"])
    b = params["safety params"]["linear cbf b"]
    alpha = params["safety params"]["alpha"]


    def k_state_constraint(x_now):
        h =  cbf(x_now)
        u_nominal = k_nominal(x_now)
        if a @ (A_euler @ x_now + B_euler @ u_nominal + x_now) + b >= 0: 
            return u_nominal
        else: 
            u_star = B_euler.T @ a.T * (- a @ (A_euler @ x_now + x_now)- b)
            u_star /= a @ B_euler @ B_euler.T @ a.T  
            return u_star


    return k_state_constraint


def get_k_discrete_time_euler_chance_state_constraint_1D_double_integrator(params, cbf, k_nominal): 
    A_euler = A_1D_double_integrator * params["time step"]
    B_euler = B_1D_double_integrator * params["time step"]
    a = np.array(params["safety params"]["linear cbf a"])
    b = params["safety params"]["linear cbf b"]
    alpha = params["safety params"]["alpha"]

    covariance = np.array(params["process noise params"]["covariance"])
    if np.sum(covariance/covariance[0,0] == np.eye(covariance.shape[0])) < np.sum(covariance.shape): 
        raise Exception("Currently the chance constraints can only handle diagonal covariances with equal values across the diagonal.")

    worst_case_unit_noise = a.T/np.linalg.norm(a) 
    chance_stddevs = scipy.stats.norm.ppf(0.5 + params["safety params"]["chance constraint probability"]/2) # Calculate Number of standard deviations of folded normal distribution for desired probability
    variance = covariance[0,0]
    if chance_stddevs >= 0:
        worst_case_noise = worst_case_unit_noise * chance_stddevs * variance
    else: 
        worst_case_noise = worst_case_unit_noise * 0 

    def k_state_constraint(x_now):
        h =  cbf(x_now)
        u_nominal = k_nominal(x_now)
        if a @ (A_euler @ x_now + B_euler @ u_nominal + x_now - worst_case_noise) + b >= 0: 
            return u_nominal
        else: 
            u_star = B_euler.T @ a.T * (- a @ (A_euler @ x_now + x_now- worst_case_noise)- b)
            u_star /= a @ B_euler @ B_euler.T @ a.T  
            return u_star

    return k_state_constraint



def get_k_discrete_time_euler_chance_cbf_1D_double_integrator(params, cbf, k_nominal): 
    A_euler = A_1D_double_integrator * params["time step"]
    B_euler = B_1D_double_integrator * params["time step"]
    a = np.array(params["safety params"]["linear cbf a"])
    b = params["safety params"]["linear cbf b"]
    alpha = params["safety params"]["alpha"]

    covariance = np.array(params["process noise params"]["covariance"])
    if np.sum(covariance/covariance[0,0] == np.eye(covariance.shape[0])) < np.sum(covariance.shape): 
        raise Exception("Currently the chance constraints can only handle diagonal covariances with equal values across the diagonal.")

    worst_case_unit_noise = a.T/np.linalg.norm(a) 
    chance_stddevs = scipy.stats.norm.ppf(0.5 + params["safety params"]["chance constraint probability"]/2) # Calculate Number of standard deviations of folded normal distribution for desired probability
    variance = covariance[0,0]
    if chance_stddevs > 0:
        worst_case_noise = worst_case_unit_noise * chance_stddevs * variance
    else: 
        worst_case_noise = worst_case_unit_noise * 0 

    def k_cbf(x_now):
        h =  cbf(x_now)
        u_nominal = k_nominal(x_now)
        if a @ (A_euler @ x_now + B_euler @ u_nominal + x_now - worst_case_noise) + b - alpha * h >= 0: 
            return u_nominal
        else: 
            u_star = B_euler.T @ a.T * (alpha * h - a @ (A_euler @ x_now + x_now - worst_case_noise)- b)
            u_star /= a @ B_euler @ B_euler.T @ a.T  
            return u_star


    return k_cbf

import cvxpy as cp 

def get_k_dtcbf_account_for_jensens(params, n, m, cbf, k_nominal, F, gap): 
    alpha = params["safety params"]["alpha"]


    def k_cbf(x_now):

        u = cp.Variable((m,1)) 
        u_des, _ = k_nominal(x_now)
        objective = cp.Minimize(cp.norm(u - u_des)**2) 
        x, rot_h = F(x_now,u)
        constraints = [cbf(x, rot_h, True) >= alpha * cbf(x_now) + gap ]
        prob = cp.Problem(objective, constraints) 
        # try: 
        try: 
            result = prob.solve(solver=cp.MOSEK)
        except: 
            result = prob.solve(solver=cp.SCS)
 
        if result != np.inf :
            u_star = u.value.reshape(4,1)
            return u_star, [[-1]] 
        else: 
            print("Using slack")
            # add slack variable 
            u = cp.Variable((m,1)) 
            slack = cp.Variable((1,1))
            slack_cost = 1
            u_des, _ = k_nominal(x_now)
            objective = cp.Minimize(cp.norm(u - u_des)**2 + slack_cost*slack**2)
            x, rot_h = F(x_now,u)
            constraints = [cbf(x, rot_h, True) + slack>= alpha * cbf(x_now) + gap ]
            prob = cp.Problem(objective, constraints) 
            try: 
                result = prob.solve(solver=cp.MOSEK)
                if result == np.inf : 
                    # controller infeasible
                    breakpoint()
            except: 
                result = prob.solve(solver=cp.SCS)

            u_star = u.value.reshape(4,1)



            return u_star, [[slack[0,0].value]] 

    return k_cbf


def get_k_dtcbf_variable_jensens(params, n, m, cbf, k_nominal, F_nom, generator, dhdx_max_eig): 
    alpha = params["safety params"]["alpha"]
    z = torch.tensor([[0.0, 0, 1]], dtype = torch.float64)

    jensen_flag = params["safety params"]["account for jensen gap"]


    def k_cbf(x_now):

        d_mean, d_cov = generator(x_now)

        jensen_gap = dhdx_max_eig * torch.trace(d_cov) * jensen_flag


        # print(torch.trace(d_cov))

        def F_adjusted(x,u):
            x_next, rot_h = F_nom(x,u) 
            if jensen_flag: 
                for i in range(3): 
                    x_next[i] += d_mean[i]
                    x_next[i+7] += d_mean[i+6]
                d_rot = pp.so3(d_mean[3:6,0]).Exp().matrix()

                rot_h += z @ d_rot @ z.T

            return x_next, rot_h

        u = cp.Variable((m,1)) 
        u_des, _ = k_nominal(x_now)
        objective = cp.Minimize(cp.norm(u - u_des)**2) 
        x, rot_h = F_adjusted(x_now,u)
        constraints = [cbf(x, rot_h, True) >= alpha * cbf(x_now) + jensen_gap ]
        prob = cp.Problem(objective, constraints) 
        # try: 
        try: 
            result = prob.solve(solver=cp.MOSEK)
        except: 
            result = prob.solve(solver=cp.SCS)
 
        if result != np.inf :
            u_star = u.value.reshape(4,1)

            if np.abs(u_star[0,0]) > 9.81 * 3: 
                u_star[0,0] = 3 * 9.81 * np.sign(u_star[0,0])

            return u_star, [[-1]] 
        else: 
            # add slack variable 
            u = cp.Variable((m,1)) 
            slack = cp.Variable((1,1))
            slack_cost = 10**8
            u_des, _ = k_nominal(x_now)
            objective = cp.Minimize(cp.norm(u - u_des)**2 + slack_cost*slack**2)
            x, rot_h = F_adjusted(x_now,u)
            constraints = [cbf(x, rot_h, True) + slack>= alpha * cbf(x_now) + jensen_gap ]
            prob = cp.Problem(objective, constraints) 
            try: 
                result = prob.solve(solver=cp.MOSEK)
                if result == np.inf : 
                    # controller infeasible
                    breakpoint()
            except: 
                result = prob.solve(solver=cp.SCS)

            u_star = u.value.reshape(4,1)

            if slack[0,0].value > 0: 
                pass
                # print("slackened controller used") # print(d_mean)
                # print(d_cov)

            if np.abs(u_star[0,0]) > 9.81 * 3: 
                u_star[0,0] = 3 * 9.81 * np.sign(u_star[0,0])

            return u_star, [[slack[0,0].value]] 

    return k_cbf



import cvxpy as cp 
from include.systems_utils.pendulum import * 

def get_k_dtcbf_pendulum(params, n, m, cbf, k_nominal, F, gap): 
    dt = params["time step"]
    alpha = params["safety params"]["alpha"]

    # Pendulum Dynamics 
    A = np.eye(2) + dt * A_continuous_pendulum
    B = dt * B_continuous_pendulum

    scaling_factor = 6**2/np.pi**2/np.sqrt(3)
    P = scaling_factor*np.array([[np.sqrt(3), 1],[1, np.sqrt(3)]])


    u = cp.Variable((1,)) 
    slack = cp.Variable((1,1))
    slack_cost = 10**11

    C0 = cp.Parameter((1,1))
    C1 = cp.Parameter((1,1))
    C2 = cp.Parameter((1,1))
    C3 = cp.Parameter((1,1),nonpos=True)
    C4 = cp.Parameter((1,1))

    objective = cp.Minimize(cp.norm(u - C0)**2 + slack_cost*slack**2)
    constraints = [C1 + C2 @ u + C3 @ u**2  + slack >= C4 ]
    prob_with_slack = cp.Problem(objective, constraints) 



    obj_no_slack = cp.Minimize(cp.norm(u - C0)**2)
    constraints = [C1 + C2 @ u + C3 @ u**2  >= C4 ]
    prob_without_slack = cp.Problem(objective, constraints)


    def k_cbf(x_now):
        grav = np.array([[0, dt*np.sin(x_now[0])[0]]]).T

        C0.value = k_nominal(x_now)
        C1.value = 1 - x_now.T @ A.T @ P @ x_now - 2 * (x_now.T @ A.T @ P @ grav) - grav.T @ P @ grav
        C2.value = - 2 * (grav.T @ P @ B) - 2 * (x_now.T @ A @ P @ B) 
        C3.value = - (B.T @ P @ B) 
        C4.value = alpha * ( 1- x_now.T @ P @ x_now ) + gap 

        try: 
            result = prob_without_slack.solve(solver=cp.MOSEK)
            if result == np.inf : 
                result = prob_with_slack.solve(solver=cp.MOSEK)
                print("slack")
                return [u.value.reshape(1,1), slack.value]
            return [u.value.reshape(1,1), np.array([[-1]])]

        except: 
            result = prob_without_slack.solve(solver=cp.SCS)
            if result == np.inf: 
                result = prob_with_slack.solve(solver=cp.SCS)
                print("slack")
                return [u.value.reshape(1,1), slack.value]
            return [u.value.reshape(1,1),np.array([[-1]])]

    return k_cbf