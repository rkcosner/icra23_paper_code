
import inspect

def convert_params_to_strings(params): 
    
    # Convert all functions to strings
    for key in params.keys():
        if isinstance(params[key], dict):
            for key2 in params[key].keys():
                if callable(params[key][key2]):
                    params[key][key2] = inspect.getsourcelines(params[key][key2])[0][0]
        elif callable(params[key]):
            params[key] = inspect.getsourcelines(params[key])[0][0]

    return params

