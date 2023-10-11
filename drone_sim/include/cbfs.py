import numpy as np
import cvxpy as cp
import pypose as pp


def setup_drone_floor_barrier():

    # See drone includes for theory

    P = np.array( [[366.15247938, 33.88556185],
                    [33.88556185, 37.12019077]])
    eigs, _ = np.linalg.eig(P)
    hessian_h_max_eig = eigs[0]/400
    lambduh = 0.1 
    constant_offset = 400 - lambduh 
    z_offset = 1.095 + 0.05 #+ 0.25
    unit_z = np.array([[0,0,1.0]]).T

    def barrier(x, rot_h = 0 , flag = False) : 

        if flag: 
            # Calculate the CBF with an adjusted rot_h rotational effect term. Necessary for working with cvxpy to include the disturbance
            diff_flat_h = constant_offset - cp.quad_form(cp.hstack([x[2] - z_offset, x[9]]), P) 
            h = (diff_flat_h + lambduh * rot_h)/400    
            return h
        else: 
            # Calculate the CBF normally 
            diff_flat_h = constant_offset - cp.quad_form(cp.hstack([x[2] - z_offset, x[9]]), P) 
            Rot = pp.SO3([x[4,0], x[5,0], x[6,0], x[3,0]]).matrix().numpy()
            rot_h = lambduh * unit_z.T @ Rot @ unit_z
            h = (diff_flat_h + rot_h)/400
            return h 


    
    return barrier, hessian_h_max_eig

if __name__ == "__main__": 
    P = np.array( [[366.15247938, 33.88556185],
                    [33.88556185, 37.12019077]])

    # unit_z R(q) unit_z  = (1 - qx^2 - qy^2) from definition of rotation matrix
    l = 0.1 
    Hessian_h = 2 / 400 * np.array([    [ P[0,0], P[0,1], 0, 0, 0, 0], 
                                        [ P[1,0], P[1,1], 0, 0, 0, 0], 
                                        [      0,      0, 0, 0, 0, 0], 
                                        [      0,      0, 0, l, 0, 0], 
                                        [      0,      0, 0, 0, -l, 0], 
                                        [      0,      0, 0, 0, 0, 0]])

    eigs, _ = np.linalg.eig(Hessian_h)

    print(eigs[0])

    breakpoint()