import numpy as np 

A_1D_double_integrator = np.array([[0,1],
                                   [0,0]])

B_1D_double_integrator = np.array([[0],
                                   [1]])

def f_1D_double_integrator(x):
    f = A_1D_double_integrator @ x
    return f

def g_1D_double_integrator(x): 
    g = B_1D_double_integrator
    return g

def k_default_1D_double_integrator(x): 
    K_prop = 0.5
    return K_prop * np.array([[-x[0,0]]])

class LinearCBF():
    def __init__(self, params): 
        self.a = np.array(params["linear cbf a"])
        self.b = params["linear cbf b"]
        self.alpha = params["alpha"]

    def get_value(self,x): 
        h = self.a @ x + self.b 
        return h.item()


