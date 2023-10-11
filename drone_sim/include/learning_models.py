
from sklearn.preprocessing import StandardScaler
import sys 
import json
import numpy as np
sys.path.append("/home/ryan/Documents/Research/Icra23/Alberts")
from cvae_model import * 


# Import model variables from JSON file
with open("model_params.json") as f:
    model_params = json.load(f)


input_scaler = StandardScaler()
target_scaler = StandardScaler()
input_scaler.mean_   = model_params["input scaler mean"]
input_scaler.var_    = model_params["input scaler var"]
input_scaler.scale_  = model_params["input scaler scale"]
target_scaler.mean_  = model_params["target scaler mean"]
target_scaler.var_   = model_params["target scaler var"]
target_scaler.scale_ = model_params["target scaler scale"]


model = CVAE(   model_params["latent dim"], 
                model_params["obs dim"],
                model_params["input dim"],
                hidden_layers=model_params["hidden layers"],
                input_scaler=input_scaler,
                target_scaler=target_scaler,
            )  

model.load_state_dict(torch.load("./py_sim_scalers/py_sim.pth"))