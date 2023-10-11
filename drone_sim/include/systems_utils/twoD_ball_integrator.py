import numpy as np 

A_2D_double_integrator = np.array([[0,0,1,0],
                                   [0,0,0,1], 
                                   [0,0,0,0], 
                                   [0,0,0,0]])

B_2D_double_integrator = np.array([[0,0],
                                   [0,0], 
                                   [1,0], 
                                   [0,1]])

def f_2D_ball(x):
    f = A_2D_double_integrator @ x
    return f

def g_2D_ball(x): 
    g = B_2D_double_integrator
    return g

def k_default_2D_ball(x): 
    u = np.zeros((2,1))
    return u 
