import numpy as np
import pickle
import os

def objective_function(X):

    pickle_dir= os.path.join('gp_samples', '1d_gp_fun')
    with open(pickle_dir, 'rb') as output:
       input_dim, f_interp_list = pickle.load(output)

    x_arr= X.T

    if input_dim == 2:
        y_arr = [np.array([f_interp(x, y) for (x, y) in zip(x_arr[0], x_arr[1])]).flatten() for f_interp in
                 f_interp_list]
    else:
        y_arr = [f_interp(x_arr.flatten()) for f_interp in f_interp_list]

    y_= np.vstack(y_arr)
    y= y_.T
    return y


objective_dim = 2
obj_dim= objective_dim
design_dim = 1
x_bound= [[1.0], [0.0]] #limits on design dimensions. upper limit list followed by lower limit list
y_bound= [[1.0, 1.0], [-1, -1]]

results_dir = './Results'

number_of_iterations = 1
confidence_ratio = 0.95 #set confidence ratio from [0, 1]

#adaptive parameters
rho= 0.5
v1 = 0.5 * np.sqrt(design_dim)
alpha= 1
h_max = 10
N_refine= 2
epsilon_percent_list= [0.5]

gp_sample= True
lengthscale_list= [0.1, 0.06]
variance_list= [0.5, 0.1]

kernel_type= 'rbf'

noisy_observations= True
noise_std_obj= 0.001
noise_std_mod= noise_std_obj


check_confidence_bounds = False
N_discard= 0