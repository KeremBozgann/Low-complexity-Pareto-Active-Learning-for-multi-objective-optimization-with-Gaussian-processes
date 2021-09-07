import numpy as np
from numpy import power
import pickle

def get_grid(x_lim,  disc, d):

    x_list = []
    for i in range(d):
        xi = np.linspace(x_lim[1][i], x_lim[0][i], disc)
        x_list.append(xi)
    X_temp = np.meshgrid(*x_list)

    X_grid = np.empty([disc ** d, d])

    for i in range(d):
        X_grid[:, i] = X_temp[i].flatten()
    del X_temp


    return X_grid

def get_grid_exact_matching_of_adaptive_grid(x_lim , hmax , d , N_refine):

    diff = np.array([ x_lim[0][i] - x_lim[1][i] for i in range(d)])
    coeff_list = np.tile(np.arange(power(N_refine , hmax)).reshape(-1 , 1) , reps = (1 , d))
    grid_list = diff / (2 * power(N_refine , hmax)) + coeff_list * diff / power(N_refine , hmax)
    temp= np.meshgrid(*grid_list.T)

    x_grid = np.zeros([power(power(N_refine , d) , hmax), d])
    for i in range(d):
        x_grid[:, i] = temp[i].flatten()
    return x_grid

def test_get_grid_exact_matching_of_adaptive_grid():
    import matplotlib.pyplot as plt

    d = 2
    x_lim = [[1] * d , [0] * d]
    N_refine = 2
    hmax = 5
    x_grid = get_grid_exact_matching_of_adaptive_grid(x_lim, hmax, d, N_refine)

    if d == 1:
        fig , ax = plt.subplots(1)
        ax.scatter(x_grid[: , 0] , np.zeros(power(power(N_refine , d) , hmax)))
        plt.show()

    elif d == 2:
        fig , ax = plt.subplots(1)
        ax.scatter(x_grid[: , 0] , x_grid[: , 1])
        plt.show()

    plt.close()

def normalize_minmax_minus_one_one(array, min, max):

    scale= max-min
    array_n= (array-min)/scale*2-1
    return array_n


def normalize_minmax_zero_one(array, min, max):
    scale = max - min
    array_n = (array - min) / scale
    return array_n


def denormalizeX(X_grid_n, scalex, minx):

    X_grid= X_grid_n*scalex+ minx

    return X_grid


def denormalizeY(Y_grid_n, scaley, miny):
    Y_grid = (Y_grid_n+1)*scaley/2+miny

    return Y_grid

import os

def create_result_dir(results_dir, dataset_name, algorithm_name):

    target_dir= results_dir +'/'+dataset_name+'/'+algorithm_name

    if not os.path.isdir(results_dir +'/'+dataset_name):
        os.mkdir(results_dir +'/'+dataset_name)


    if not os.path.isdir(target_dir):
        os.mkdir(target_dir)


    return target_dir

def pickle_read(file_name):
    try:
        with open(file_name, 'rb') as f:
            return pickle.load(f)
    except Exception as ex:
        print(ex)
        return None

def pickle_save(file_name, data):
    try:
        with open(file_name, 'wb') as f:
            return pickle.dump(data, f)
    except Exception as ex:
        print(ex)
        return None