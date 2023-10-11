import numpy as np

from include.systems_utils.oneD_double_integrator import * 
from include.systems_utils.twoD_ball_integrator import* 
from include.systems_utils.steinhardt_1d_sys import * 
from include.systems_utils.pendulum import * 
from include.systems_utils.drone import * 


system_types = {
    "1D_Double_Integrator" : {
        "n state size" : 2, 
        "m input size" : 1,
        "f" : f_1D_double_integrator, 
        "g" : g_1D_double_integrator, 
        "initial condition" : np.array([[2],[0]]), 
        "k default" : k_default_1D_double_integrator,
        "state labels" : [r"$x$", r"$v$"],
        "input labels" : [r"$a$"],
        "slack labels" : ["slack"],
    }, 
    "2D_Ball" : {
        "n state size" : 4,
        "m input size" : 2, 
        "f" : f_2D_ball, 
        "g" : g_2D_ball, 
        "initial condition" : np.array([[0.1, 2, 0, -1]]).T, 
        "k default" : k_default_2D_ball, 
        "state labels" : [r"$x$", r"$y$", r"$\dot{x}$", r"$\dot{y}$"], 
        "input labels" : [r"$a_x$", r"$a_y$"], 
        "slack labels" : ["slack"],
    }, 
    "1D_steinhardt" : {
        "n state size" : 1,
        "m input size" : 1, 
        "f" : f_steinhardt_1D, 
        "g" : g_steinhardt_1D, 
        "get F" : get_F_steinhardt_1D, 
        "initial condition" : np.array([[0]]).T, 
        "k default" : k_steinhardt_1D, 
        "state labels" : [r"$x$"], 
        "input labels" : [r"$u$"], 
        "slack labels" : ["slack"],
    }, 
    "pendulum" : {
        "n state size" : 2,
        "m input size" : 1, 
        "f" : f_pendulum, 
        "g" : g_pendulum, 
        "get F" : get_F_pendulum, 
        "initial condition" : np.array([[0.0, 0.0]]).T, 
        "k default" : k_pendulum, 
        "state labels" : [r"$\theta$", r"$\dot{\theta}$"], 
        "input labels" : [r"$u$"], 
        "slack labels" : ["slack"],
    }, 
    "drone" : {
        "n state size" : 10,
        "m input size" : 4, 
        "p disturbance size" : 9,
        "f" : np.NaN, 
        "g" : np.NaN, 
        "get F" : get_F_drone, 
        "get F eul": get_euler_F_drone,
        "initial condition" : np.array([[0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]).T, 
        "k default" : k_fall, 
        "SE(3) controller" : k_SE3,
        "state labels" : [r"$x$", r"$y$", r"$z$", r"$q_w$",  r"$q_x$",  r"$q_y$",  r"$q_z$",  r"$vx$",  r"$vy$",  r"$vz$"], 
        "input labels" : [r"$a_{thrust}$",  r"$\omega_x$",  r"$\omega_y$",  r"$\omega_z$", "slack"], 
        "slack labels" : ["slack"],
        "add disturbance": drone_add_disturbance, 
    }, 
}



