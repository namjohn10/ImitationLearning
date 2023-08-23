import torch
import pickle5 as pickle

## Checkpoint Path
path = None 

state = None
with open(path, "rb") as f:
    state = pickle.load(f)
    print(state["metadata"])

    state["muscle_optimizer"] = None  
    
with open(path + '_revised', 'wb') as f:
    pickle.dump(state, f)
