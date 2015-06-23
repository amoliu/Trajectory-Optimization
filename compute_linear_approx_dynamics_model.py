import numpy as np

def compute_linear_approx_dynamics_model(f, curr_X, curr_U, dt, eps=pow(10, -5)): 
    T = curr_X.shape[1]
    assert curr_X.shape[1] == curr_U.shape[1]

    import pdb; pdb.set_trace() 


    
