import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

def highdim_feature(model, x, delta_y, pca_matrix=None):
    # estimate the delta_x required at x to change output embedding by delta_y
    
    input_x = torch.Tensor(x).float()    
    input_x.requires_grad = True
    
    if pca_matrix is not None:
        output = model(torch.matmul(torch.Tensor(pca_matrix).float(), input_x))
    
    else:
        output = model(input_x)
    
    #print(input_x.shape)
    
    # calculate Jacobian matrix of embedding w.r.t. input data
    df_mat = [torch.autograd.grad(outputs=out, inputs=input_x, retain_graph=True)[0] for i, out in enumerate(output)]
    df_mat = [df_mat[i].detach().numpy() for i in range(len(df_mat))]
    df_mat = np.array(df_mat)
    
    #print(df_mat.shape)
    #print(len(x))
    assert df_mat.shape[-1] == len(x), 'Error: gradient w.r.t. x does not match shape of x'
    
    
    sol = np.linalg.inv(df_mat[:, :len(delta_y)]).dot(delta_y).squeeze()
    sol = list(sol) + [0]*(len(x)-len(delta_y))
    
    assert len(sol) == len(x)
    sol = np.array(sol)

    delta_x = 0
    for i in range(len(df_mat)):
        delta_x += sol.dot(df_mat[i])/(df_mat[i].dot(df_mat[i])) * df_mat[i]
        
    new_x = x + delta_x
    return delta_x, new_x, df_mat 
