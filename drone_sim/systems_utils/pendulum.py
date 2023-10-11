import numpy as np 
import cvxpy as cp

A_continuous_pendulum = np.array([[0, 1], 
                            [0, 0]])

B_continuous_pendulum = np.array([[0],
                            [1]])

def f_pendulum(x):
    f = A_continuous_pendulum @ x
    f += np.array([[0], [-np.sin(x[0])]]) 
    return f

def g_pendulum(x): 
    g = B_continuous_pendulum
    return g

def get_F_pendulum(time_step, process_noise): 
    
    def F(x, u): 

        x_next = x 
        x_next = x_next + time_step * A_continuous_pendulum @ x
        x_next = x_next + time_step * np.array([[0.0, np.sin(x[0])[0]]]).T  
        x_next = x_next + time_step * B_continuous_pendulum @ u 
        # x_next = x_next + (np.sqrt(time_step)*process_noise["get sample"]()*process_noise["on/off"]).flatten("C")        

        return x_next

    return F 

def k_pendulum(x): 
    u = np.zeros((1,1))
    return u 
