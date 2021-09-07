import matplotlib.pyplot as plt
plt.ioff()
import numpy as np
import os
import h5py
import sys
import experiment_setup
from loss_calculations import get_pareto_front_set , get_pareto_front_set_special_m2
from util import get_grid_exact_matching_of_adaptive_grid

def get_adaptive_front_and_evaluations(adaptive_path, eps_percent_adapt, iter_num):
    #load adaptive
    with h5py.File(adaptive_path, 'r') as hf:
        adaptive_pareto_set= np.array(hf.get('epsilon_percent_{}_iter_{}_Pareto_set_normalized'.format(eps_percent_adapt, iter_num-1)))
        adaptive_pareto_front = np.array(hf.get('epsilon_percent_{}_iter_{}_Pareto_front_normalized'.format(eps_percent_adapt,  iter_num-1)))
        XT_adapt = np.array(hf.get('epsilon_percent_{}_iter_{}_XT'.format(eps_percent_adapt,  iter_num-1)))
        YT_adapt = np.array(hf.get('epsilon_percent_{}_iter_{}_YT'.format(eps_percent_adapt,  iter_num -1)))
    return adaptive_pareto_set,  adaptive_pareto_front , XT_adapt , YT_adapt


def get_epsilon_coverage_rate(predicted_pareto_front, approximated_true_pareto_front, epsilon, alg= None):

    if not alg== 'usemo':

        _, predicted_pareto_front_minus = get_pareto_front_set_special_m2(np.random.rand(*predicted_pareto_front.shape),
                                                                    -predicted_pareto_front, get_indeces=False)
        predicted_pareto_front= - predicted_pareto_front_minus.copy()

    covered = np.zeros([1,approximated_true_pareto_front.shape[0]], dtype= 'bool')
    for i in range(approximated_true_pareto_front.shape[0]):
        pareto_front_i = approximated_true_pareto_front[i, :].reshape(-1, 1)

        for j in range(predicted_pareto_front.shape[0]):
            predicted_pareto_foront_n_j =predicted_pareto_front[j, :].reshape(-1, 1)

            if np.all(predicted_pareto_foront_n_j + epsilon >= pareto_front_i):
                covered[0, i] = True
                break
        # for debugging
        if covered[0, i]== False:
            pass

    return np.mean(covered)


def get_epsilon_acc_rate(predicted_pareto_front, approximated_true_pareto_front, epsilon,  usemo= False):

    accurate= np.ones([1, predicted_pareto_front.shape[0]])

    if usemo:
        _, predicted_pareto_front= get_pareto_front_set_special_m2( np.random.rand(*predicted_pareto_front.shape),predicted_pareto_front, get_indeces=False)
    for i in range(predicted_pareto_front.shape[0]):
        obj_i= predicted_pareto_front[i, :].reshape(-1,1)

        for j in range(approximated_true_pareto_front.shape[0]):
            pareto_j= approximated_true_pareto_front[j,:].reshape(-1,1)

            if np.all(pareto_j-epsilon>=obj_i):
                accurate[0, i]= 0
                break

    return np.mean(accurate)

def get_hypervolume(predicted_pareto_front,ref, alg):
    from pygmo import hypervolume

    if not alg== 'usemo':

        _, predicted_pareto_front_minus = get_pareto_front_set_special_m2(np.random.rand(*predicted_pareto_front.shape),
                                                                    -predicted_pareto_front, get_indeces=False)
        predicted_pareto_front= - predicted_pareto_front_minus.copy()

    hv= hypervolume(-predicted_pareto_front)
    hypv= hv.compute(-ref[0, :])
    return hypv


def bar_metrics(cov, acc,func_name):

    X = np.array([0.0, 0.3])
    fig, ax = plt.subplots(1)
    ax.set_title('Adaptive epal')
    ax.bar(X[0], acc, color='tab:blue', width=0.2, label='Acc. Rat.')
    ax.bar(X[1], cov, color='tab:red', width=0.2, label='Cov. Rat.')
    ax.legend()

    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['font.family'] = 'Calibri'
    plt.rcParams['xtick.major.pad'] = '5'
    plt.rcParams['ytick.major.pad'] = '5'
    plt.savefig(os.path.join('Results', func_name , 'performance_metrics.pdf'))
    plt.close()

def plot_pareto_front(pred_front, true_front, evaluations, func_name):

    fig, ax = plt.subplots(1)
    ax.scatter(true_front[:, 0], true_front[:, 1], label='True Pareto front', alpha= 0.3,  marker = "o",color='black', s = 9)
    ax.scatter(evaluations[:, 0], evaluations[:, 1], label='Evaluations', alpha=0.4, marker='+',color='darkblue', s=1)
    ax.scatter(pred_front[:, 0], pred_front[:, 1], label='Predicted Pareto front', alpha= 1,  marker=".", color='crimson', s=2)

    ax.grid()
    legend= ax.legend()
    legend.get_frame().set_alpha(None)

    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)
    ax.set_title(f'Adaptive ePAL predicted pareto front')
    ax.set_facecolor('white')
    ax.grid(b=True, which='major', color='black', alpha=0.5, linestyle='--', linewidth=0.5)

    ax.spines['bottom'].set_color('black')
    ax.spines['top'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.spines['right'].set_color('black')

    ax.set_xlabel(r'$f_1$')
    ax.set_ylabel(r'$f_2$')

    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['font.family'] = 'Calibri'
    plt.rcParams['xtick.major.pad'] = '5'
    plt.rcParams['ytick.major.pad'] = '5'
    plt.savefig(os.path.join('Results', func_name, 'predicted_front.pdf'))
    plt.close()

def plot_results(func_name, kern_type):
    from experiment_setup import objective_function

    sys.path.append(os.path.join('..', 'example_setups'))

    dim = experiment_setup.design_dim
    h_disc= experiment_setup.h_max

    path = func_name


    iter_num = experiment_setup.number_of_iterations
    if  iter_num== 1:

        path_adaptive = os.path.join('Results', path, 'adaptive', 'exp.h5')

    else:

        path_adaptive= os.path.join('adaptive', 'exp.h5')

    ls_list= experiment_setup.lengthscale_list
    var_list= experiment_setup.variance_list
    obj_dim = experiment_setup.obj_dim
    eps_percent = experiment_setup.epsilon_percent_list[0]
    N_refine= experiment_setup.N_refine
    epsilon= eps_percent/100*2
    x_lim = [[1] *dim , [0] * dim]

    adaptive_pareto_set, adaptive_pareto_front, XT_adapt, YT_adapt= get_adaptive_front_and_evaluations(path_adaptive, eps_percent, iter_num)


    x = get_grid_exact_matching_of_adaptive_grid(x_lim , h_disc , dim , N_refine)
    objectives= objective_function(x)

    if obj_dim == 2:
        pareto_set , pareto_front = get_pareto_front_set_special_m2(x, objectives, get_indeces=False)
    else:
        pareto_set , pareto_front = get_pareto_front_set(x , objectives)

    cov_adapt = get_epsilon_coverage_rate(adaptive_pareto_front, pareto_front, epsilon, alg='adapt')

    acc_adapt = get_epsilon_acc_rate(adaptive_pareto_front, pareto_front, epsilon)

    delt = np.zeros([obj_dim, 1])
    for i in range(obj_dim):
        if kern_type == 'matern52':
            delt[i, 0] = np.sqrt(10) * np.sqrt(var_list[i]) / ls_list[i] * (np.sqrt(dim)* (x[1, 0]- x[0, 0])/2)
        elif kern_type =='matern32':
            delt[i, 0] = np.sqrt(6) * np.sqrt((var_list[i]) / ls_list[i] * (np.sqrt(dim) * (x[1, 0] - x[0, 0]) / 2))
        elif kern_type == 'sqrexp':
            delt[i, 0] = np.sqrt(var_list[i]) / ls_list[i] * (np.sqrt(dim) * (x[1, 0] - x[0, 0]) / 2)

    ref= np.zeros([1, obj_dim])
    for i in range(obj_dim):
        ref[0, i]= np.min(objectives[:, i]) - delt[i, 0]


    hypv_adapt = get_hypervolume(adaptive_pareto_front, ref, alg='adapt')

    print('hypervolume: ', hypv_adapt)

    bar_metrics(cov_adapt, acc_adapt,  func_name)

    plot_pareto_front(adaptive_pareto_front, pareto_front, YT_adapt, func_name)


