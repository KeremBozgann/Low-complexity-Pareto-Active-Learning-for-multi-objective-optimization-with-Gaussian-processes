import numpy as np
import pickle
import os

temp_dir = os.getcwd()
file_dir = os.path.dirname(__file__)
os.chdir(os.path.dirname(__file__))

if temp_dir == file_dir:
    pickle_dir = os.path.join('gp_samples', 'interp_1d_sqrexp')
else:
    pickle_dir = os.path.join('..', 'gp_samples', 'interp_1d_sqrexp')

with open(pickle_dir, 'rb') as output:
    f_interp_list = pickle.load(output)

os.chdir(temp_dir)

def objective_function(X):

    x_arr= X.T

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
alpha= 1
v1 = 0.5 * np.sqrt(design_dim)
h_max = 10
N_refine= 2
epsilon_percent_list= [0.5]


gp_sample= True
lengthscale_list= [0.2, 0.2]
variance_list= [0.6, 0.6]


kernel_type= 'rbf'

noisy_observations= True
noise_std_obj= 0.001
noise_std_mod= noise_std_obj


check_confidence_bounds = False
N_discard= 0