import numpy as np 

A_steinhardt_1D = np.array([[1]])

B_steinhardt_1D = np.array([[1]])

def f_steinhardt_1D(x):
    f = A_steinhardt_1D @ x
    return f

def g_steinhardt_1D(x): 
    g = B_steinhardt_1D
    return g

def get_F_steinhardt_1D(time_step, process_noise): 
    
    def F(x, u): 
        x_next = x + 2 + time_step * u 

        return x_next

    return F 

def k_steinhardt_1D(x): 
    # u = np.zeros((1,1))
    u = 100*(-2 -x)*np.ones((1,1))
    return u 
