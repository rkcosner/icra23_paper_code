import numpy as np 
import cvxpy as cp
import pypose as pp
import torch 

drone_mass = 1.25 # kg 
grav = 9.81 # m/sec^2 
z = torch.tensor([0.0, 0, 1]).double()

def convert2pypose(x):
    # Parse State
    pos = torch.tensor(x[:3,0], dtype=torch.float64)
    rot = pp.SO3([x[4,0], x[5,0], x[6,0], x[3,0]]).double()
    vel = torch.tensor(x[7:,0], dtype=torch.float64)

    return pos, rot, vel

def convertFrompypose(new_pos, new_rot, new_vel):
    x_next = new_pos[:].tolist()
    x_next.extend([new_rot[3], new_rot[0], new_rot[1], new_rot[2]])
    x_next.extend(new_vel[:].tolist())
    x_next = np.array([x_next]).T

    return x_next


def get_euler_F_drone(time_step): 
    # Ignoring the manifold, returning a rotation matrix


    def F(x, u): 

        pos, rot, vel = convert2pypose(x) 
        rot_mat = rot.matrix()

        Omega = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 0]])

        pos_next = pos + time_step * vel 
        rot_h = z.T @ rot_mat @ z + time_step / 2 * z.T @ Omega @ u[1:,0]
        vel_next = vel + time_step * (rot @ z * u[0,0] - grav * z) 

        x_next = [pos_next[0].item(), pos_next[1].item(), pos_next[2].item(), 0,0,0,0, vel_next[0:1], vel_next[1:2], vel_next[2:3]]

        return x_next, rot_h

    return F


def F_drone_variable_dt(x, u, time_step): 

    pos, rot, vel = convert2pypose(x)

    # Gather Input
    thrust_acc, angular_rates = u[0], u[1:]
    thrust_acc = torch.tensor(thrust_acc).double()
    angular_rates = torch.tensor(angular_rates.reshape(3,)).double()
    
    # Update
    new_pos = pos + time_step * vel
    new_rot =  rot + time_step * angular_rates        
    new_vel = vel + (time_step) * (thrust_acc * (rot @ z) - grav * z)

    # Package
    x_next = convertFrompypose(new_pos, new_rot, new_vel)

    return x_next

def get_F_drone(time_step, process_noise): 
    
    def F(x, u): 
        
        pos, rot, vel = convert2pypose(x)

        # Gather Input
        thrust_acc, angular_rates = u[0], u[1:]
        thrust_acc = torch.tensor(thrust_acc).double()
        angular_rates = torch.tensor(angular_rates.reshape(3,)).double()
        
        # Update
        new_pos = pos + time_step * vel
        new_rot =  rot + time_step * angular_rates        
        new_vel = vel + (time_step) * (thrust_acc * (rot @ z) - grav * z)

        # Package
        x_next = convertFrompypose(new_pos, new_rot, new_vel)

        return x_next


    return F 

# SE(3) controller, adjust these gains if needed 
kp = 2e1
kv = 1e1
kR = 1e1
p_des = torch.tensor([0., 0., 0.]).double()
b1_des = torch.tensor([1., 0., 0.]).double()
def k_SE3(x): 

    # Parse State
    pos, rot, vel = convert2pypose(x)

    e_p = pos - p_des
    e_v = vel

    acc_des = -(kp * e_p + kv * e_v - drone_mass * grav * z)
    thrust = (acc_des * (rot.Act(z))).sum(-1, keepdim=True)
    
    b3_des = acc_des / torch.linalg.norm(acc_des, dim=-1, keepdim=True)
    b2_des = torch.linalg.cross(b3_des, b1_des)
    b2_des = b2_des / torch.linalg.norm(b2_des, dim=-1, keepdim=True)
    b1_des_proj = torch.linalg.cross(b2_des, b3_des)

    R_des = pp.mat2SO3(torch.stack([b1_des_proj, b2_des, b3_des], dim=-1))


    e_R = ((R_des.Inv() * rot).matrix() - (rot.Inv() * R_des).matrix())[..., [2, 0, 1], [1, 2, 0]]
    
    thrust_acc = thrust / drone_mass
    w_des = -kR * e_R
        
    u = np.array([[thrust_acc.item(), w_des[0].item(), w_des[1].item(), w_des[2].item()]]).T 
    
    return  u, [[-1]]

def k_fall(x): 
    u = np.array([[0,0,0,0]]).T 

    return u, [[-1]]


def drone_add_disturbance(x, disturbance): 
    pos, rot, vel = convert2pypose(x) 
    
    dpos = torch.tensor(disturbance[ :3, 0], dtype = torch.float64)
    drot = torch.tensor(disturbance[3:6, 0], dtype = torch.float64)
    dvel = torch.tensor(disturbance[6:9, 0], dtype = torch.float64)

    pos = pos + dpos 
    rot = rot + drot 
    vel = vel + dvel

    x_next = convertFrompypose(pos, rot, vel)


    return x_next 