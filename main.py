#change function name in the "func_list" variable below to initate a different experiment from example_setups directory
from adaptive_cont import adaptive_epal_cont
import os
import pathlib
import numpy as np
from plots import plot_results, plot_hypervolume


def write_results(dir, sampling_matrix, eps_percent_list, number_of_iterations):
    file_path = os.path.join(dir, 'results.txt')

    with open(file_path, 'a') as f:

        f.write('average_samplings:' + str(np.mean(sampling_matrix, axis=0)) + '\n')
        f.write('epsilon_percentage_list:' + str(eps_percent_list) + '\n')
        f.write('number of iterations:' + str(number_of_iterations) + '\n')


def run_experiments():

        import importlib

        import sys
        import shutil
        # sys.path.append('.\\example_setups')

        example_setups_dir = os.path.join(os.getcwd(), 'example_setups')
        sys.path.append(example_setups_dir)


        # func_list = ['sqrexp_1d']
        func_list = ['sqrexp_2d']
        # func_list = ['sqrexp_1d', 'matern52_1d', 'sqrexp_2d']

        kernel_list= ['sqrexp']
        # kernel_list = ['sqrexp', 'matern52', 'sqrexp', 'matern52']


        print('func_list')
        print(func_list)

        for i, func_name in enumerate(func_list):
            kern_type= kernel_list[i]
            print(func_name)
            setup_name = 'experiment_setup_{}'.format(func_name)

            original = r'example_setups/experiment_setup_{}.py'.format(func_name)
            target = r'experiment_setup.py'

            shutil.copyfile(original, target)

            exec('import ' + setup_name)
            importlib.reload(eval(setup_name))
            exec('import ' + setup_name)

            import experiment_setup
            importlib.reload(experiment_setup)
            import experiment_setup

            print('hmax')
            print(experiment_setup.h_max)
            print()

            save_dir_adaptive_cont = os.path.join('Results', func_name , 'adaptive')
            pathlib.Path(save_dir_adaptive_cont).mkdir(parents=True, exist_ok=True)

            sampling_matrix_adapt_cont, refine_matrix_adaptive_cont = adaptive_epal_cont(
                save_dir_adaptive_cont)
            write_results(save_dir_adaptive_cont, sampling_matrix_adapt_cont, experiment_setup.epsilon_percent_list, experiment_setup.number_of_iterations)

            if experiment_setup.obj_dim == 2:
                plot_results(func_name, kern_type)

            path_h5py  = os.path.join(save_dir_adaptive_cont,'exp.h5')
            num_eval = sampling_matrix_adapt_cont[0,0]# number of evaluations in the first iteration
            eps_perc = experiment_setup.epsilon_percent_list[0]
            iteration =0
            plot_hypervolume(path_h5py, int(num_eval), eps_perc, iteration, func_name)

run_experiments()