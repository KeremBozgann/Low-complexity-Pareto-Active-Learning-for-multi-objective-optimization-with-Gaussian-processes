#change function name to initate a different experiment from example_setups directory
from adaptive_cont import adaptive_epal_cont
import os
import pathlib
import numpy as np
from plots import plot_results


def write_results(dir, sampling_matrix, eps_percent_list, number_of_iterations):
    with open(dir + '/' + 'results.txt', 'a') as f:

        f.write('average_samplings:' + str(np.mean(sampling_matrix, axis=0)) + '\n')
        f.write('epsilon_percentage_list:' + str(eps_percent_list) + '\n')
        f.write('number of iterations:' + str(number_of_iterations) + '\n')


def run_experiments():

        import importlib

        import sys
        import shutil
        sys.path.append('.\\example_setups')

        func_list = ['sqrexp_1d']
        kernel_list= ['sqrexp']
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

run_experiments()