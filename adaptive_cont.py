import numpy as np
from util import get_grid, denormalizeX, pickle_save
import h5py
import experiment_setup
from experiment_setup import objective_function
from pareto_set_operations import discarding, epsilon_covering, refine_nodes
import os
from confidence_calculation_related_operations import  calculate_vh , update_objective_cells, gp_model
from numpy import log , power , pi


def create_vec(d):

    Vec = np.zeros([1, 2*d])
    Vec[:, [2*i for i in range(d)]] = 1


    return Vec


def initialize_sets(epsilon_percent):

    set_dict= {}
    set_dict['m']= experiment_setup.objective_dim
    set_dict['d']= experiment_setup.design_dim
    set_dict['h_max']= experiment_setup.h_max

    if experiment_setup.gp_sample:
        # set_dict['epsilon'] =  np.ones([1, set_dict['m']]) * np.sqrt(experiment_setup.variance) * 2 * epsilon_percent / 100
        set_dict['epsilon']  = np.ones([1, set_dict['m'] ]) * 2 * epsilon_percent / 100

    else:
        set_dict['epsilon']  = np.ones([1, set_dict['m'] ]) * 2 * epsilon_percent / 100

    set_dict['design_cells_nt'] = create_vec(set_dict['d'])
    set_dict['design_cells_pt'] = np.zeros([0, 2*set_dict['d']])
    set_dict['design_cells']= np.append(set_dict['design_cells_nt'], set_dict['design_cells_pt'], axis= 0)

    set_dict['objective_cells_nt']= create_vec(set_dict['m'])
    set_dict['objective_cells_pt']= np.zeros([0, 2*set_dict['m']])
    set_dict['objective_cells']= np.append(set_dict['objective_cells_nt'], set_dict['objective_cells_pt'], axis= 0)

    set_dict['pt'] = np.zeros([0, set_dict['d']])
    set_dict['nt'] = np.ones([1 , set_dict['d']])*0.5
    set_dict['points']= np.append(set_dict['nt'], set_dict['pt'], axis=0)

    set_dict['h']= np.zeros([1,1]) #depth of the nodes
    set_dict['Vh'] = np.ones([1, set_dict['m']])

    set_dict['XT'] = np.zeros([0, set_dict['d']])
    set_dict['YT']= np.zeros([0, set_dict['m']])

    return set_dict

def test_initialize_sets():
    set_dict= initialize_sets(5)
    print(set_dict)


def find_cell_with_maximum_objective_radius_unoptimized(objective_cells, m):

    objective_cells_l= objective_cells[:, [2*i+1 for i in range(m)]]
    objective_cells_u= objective_cells[:, [2*i for i in range(m)]]

    square_dist= np.zeros([objective_cells.shape[0], 1])

    for i in range(m):
        dist_i = (objective_cells_u[:, i].reshape(-1, 1) - objective_cells_l[:, i].reshape(-1, 1)) ** 2
        square_dist = square_dist + dist_i

    dist = np.sqrt(square_dist)

    max_index = np.argmax(dist)

    return max_index


def find_cell_with_maximum_objective_radius(objective_cells, m ,  return_max_value = False):

    objective_cells_l = objective_cells[:, [2 * i + 1 for i in range(m)]]
    objective_cells_u = objective_cells[:, [2 * i for i in range(m)]]

    if return_max_value:
        arg_max_rad = np.argmax(np.linalg.norm(objective_cells_u - objective_cells_l , axis = 1)[:])
        return  arg_max_rad , (objective_cells_u[arg_max_rad] - objective_cells_l[arg_max_rad]).reshape(1 , -1)
    else:
        return np.argmax(np.linalg.norm(objective_cells_u - objective_cells_l , axis = 1)[:])


def evaluate_design(chosen_index, points, XT, YT, scalex, scaley, minx, miny, noise_obj_std):
    x_evaluate_n = np.zeros([1, experiment_setup.design_dim])

    x_evaluate_n[0, :] = points[chosen_index, :]

    x_evaluate = x_evaluate_n * scalex + minx
    # yt= input('Enter objective values obtained from evaluation at the design point {} as list :'.format(x_evaluate))

    if noise_obj_std is None:
        yt = objective_function(x_evaluate)
    else:
        yt = objective_function(x_evaluate)+ np.random.normal(0.0, noise_obj_std, (1, experiment_setup.objective_dim))

    yt_n = (yt - miny) / scaley * 2 - 1

    XT = np.append(XT, x_evaluate_n, axis=0)
    YT = np.append(YT, yt_n, axis=0)
    print()
    print('evaluated design point:')
    print(x_evaluate)
    print()
    print('objective value')
    print(yt)
    print()

    return XT, YT, x_evaluate_n, yt_n


def get_normalization_parameters():

    minx = np.array(experiment_setup.x_bound[1]).reshape(1, -1)
    maxx = np.array(experiment_setup.x_bound[0]).reshape(1, -1)

    miny = np.array(experiment_setup.y_bound[1]).reshape(1, -1)
    maxy = np.array(experiment_setup.y_bound[0]).reshape(1, -1)

    scalex= maxx- minx
    scaley= maxy- miny

    return minx, maxx, scalex, miny, maxy, scaley


def check_offline_objective_cells(objective_cells, dataY):
    N_total_data= dataY.shape[0]
    count= 0
    for ind_dat in range(dataY.shape[0]):
        y=dataY[ind_dat, :].reshape(1,-1)
        cell= objective_cells[ind_dat, :].reshape(1,-1)
        cell_l= [cell[0, i] for i in range(cell.shape[1]) if i%2==1]
        cell_u= [cell[0, i] for i in range(cell.shape[1]) if i%2==0]

        if np.all(y>= cell_l) and np.all(y<= cell_u):
            count+=1

    print('number of points that inside confidence rectangle:')
    print(count)
    print('total number of points')
    print(N_total_data)

def test_check_offline_objective_cells():
    objective_cells= np.ones([10, 4])
    objective_cells[:, 1]= np.zeros([10, ])
    objective_cells[:, 3]= np.zeros([10, ])
    Y= np.ones([10, 2])*(-1)
    check_offline_objective_cells(objective_cells,Y)
    return

def plot_points_and_cells_epal(points, objective_cells, design_cells, mean):
    import matplotlib.pyplot as plt
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Rectangle

    fig, ax = plt.subplots(1)  # design points and design cells plot
    edgecolor = 'red'
    alpha = 0.5
    rects = []
    ax.scatter(points[:, 0], points[:, 1], color='blue')
    for i in range(design_cells.shape[0]):
        cell = design_cells[i, :].reshape(1, -1)
        rects.append(Rectangle((cell[0, 1], cell[0, 3]), cell[0, 0] - cell[0, 1], cell[0, 2] - cell[0, 3]))
    pc = PatchCollection(rects, alpha=alpha, edgecolor=edgecolor)
    ax.add_collection(pc)

    plt.show()

    fig, ax = plt.subplots(1)  # objective points and objective cell plot
    edgecolor = 'red'
    alpha = 0.5
    rects = []
    ax.scatter(mean[:, 0], mean[:, 1], color='blue')
    for i in range(objective_cells.shape[0]):
        cell = objective_cells[i, :].reshape(1, -1)
        rects.append(Rectangle((cell[0, 1], cell[0, 3]), cell[0, 0] - cell[0, 1], cell[0, 2] - cell[0, 3]))
    pc = PatchCollection(rects, alpha=alpha, edgecolor= edgecolor)
    ax.add_collection(pc)

    plt.show()

def check_max_depth(flag_hmax, h, h_max, set_dict):

    if flag_hmax == 0:

        # If hmax is reached for all points in (St U Pt) , Set Vh= 0 for all points. After this, algorithm works like e-PAL
        if np.all(h == h_max):
            set_dict['Vh'] = np.zeros(set_dict['Vh'].shape)
            # Set.N_set, Set.P_set = Construct_Sets_After_Max_Depth_Reached(dat.X, Set)
            flag_hmax = 1
            print('h_max reached')

    return flag_hmax, set_dict


def check_beta_confidence_intervals(mean, std, beta, obj_dim, d, objective_function, x_lim, points, print_true= False):
    if d==1:
        disc= 8001
        x_test=  get_grid(x_lim, disc, d)
        f_test= objective_function(x_test)
    elif d==2:
        disc= 101
        x_test=  get_grid(x_lim, disc, d)
        f_test= objective_function(x_test)
    f_points= objective_function(points)

    conf_ratio_list = []
    x_out_bound_list = []
    f_out_bound_list = []

    diff_obs_list = []
    beta_sqrt = np.sqrt(beta)

    for j in range(obj_dim):
        lower_bound = mean[:, j] - beta_sqrt[j, 0] * std[:, j]
        upper_bound = mean[:, j] + beta_sqrt[j, 0] * std[:, j]
        conf_ratio_list.append(
            np.mean(np.logical_and(mean[:, j] - beta_sqrt[j, 0] * std[:, j] <= f_points[:, j],
                                   mean[:, j] + beta_sqrt[j, 0] * std[:, j] >= f_points[:, j])))
        bool_out_bound = np.invert(
            np.logical_and(mean[:, j]  - beta_sqrt[j, 0] *std[:, j]<= f_points[:, j],
                           mean[:, j] + beta_sqrt[j, 0] * std[:, j]>= f_points[:, j]))
        x_out_bound = points[bool_out_bound]
        ind_out_bound = np.arange(mean[:, j] .shape[0])[bool_out_bound]
        f_out_bound = f_points[:, j][bool_out_bound]
        x_out_bound_list.append(x_out_bound)
        f_out_bound_list.append(f_out_bound)

        ind1 = np.where(lower_bound[ind_out_bound] > f_out_bound)[0]
        ind2 = np.where(upper_bound[ind_out_bound] < f_out_bound)[0]
        diff_obs1 = lower_bound[ind_out_bound][ind1] - f_out_bound[ind1]
        diff_obs2 = f_out_bound[ind2] - upper_bound[ind_out_bound][ind2]
        diff_obs = np.append(diff_obs1, diff_obs2)

        if print_true:
            if len(diff_obs) != 0:
                print(f'maximum difference objective{j}', np.max(diff_obs))
            else:
                print(f'maximum difference objective{j}', 0)

        diff_obs_list.append(diff_obs)

    if print_true:
        print('confidence ratio', conf_ratio_list)

    return conf_ratio_list, diff_obs_list

def adaptive_epal_cont(save_dir):

    epsilon_percantage_list= experiment_setup.epsilon_percent_list

    #assumption 1 parameters
    rho= experiment_setup.rho
    alpha= experiment_setup.alpha
    v1 = experiment_setup.v1
    delta= 1- experiment_setup.confidence_ratio
    h_max= experiment_setup.h_max

    N_refine= experiment_setup.N_refine

    N_iter= experiment_setup.number_of_iterations

    refinements= np.zeros([N_iter, len(epsilon_percantage_list)])
    samplings= np.zeros([N_iter, len(epsilon_percantage_list)])

    minx, maxx, scalex, miny, maxy, scaley= get_normalization_parameters()
    x_lim= experiment_setup.x_bound

    kernel_type= experiment_setup.kernel_type
    if experiment_setup.noisy_observations:

        noise_obj_std= np.ones([1,experiment_setup.objective_dim])*experiment_setup.noise_std_obj
        noise_mod_std= np.ones([1,experiment_setup.objective_dim])*experiment_setup.noise_std_mod

    else:
        noise_obj_std= None
        noise_mod_std= None

    flag_excess= False #if number of points are too many, terminate

    check_confidence_bounds= experiment_setup.check_confidence_bounds
    for epsilon_index, epsilon_percentage in enumerate(epsilon_percantage_list):
        print()
        print('epsilon percantage')
        print(epsilon_percentage)
        print()

        for iteration in range(N_iter):
            np.random.seed(iteration+1)

            print()
            print('iteration:')
            print(int(iteration+1))
            print()

            flag_hmax= 0 #set flag to true if h_max is reached by all the nodes in (St U Pt)
            number_of_evaluations= 0
            number_of_evaluations_prev_write= 0
            number_of_refinements= 0

            # use known hyperparameters if objectives are gp samples
            gp_mod= gp_model(input_dim= experiment_setup.design_dim, obj_dim= experiment_setup.objective_dim,
                             kernel_type= kernel_type, noisy_observations= experiment_setup.noisy_observations, gp_sample= experiment_setup.gp_sample,
                             noise_mod_std= noise_mod_std)

            print()
            print('printing learned kernels')
            print()

            for i in range(len(gp_mod.kernel_list)):
                print(gp_mod.kernel_list[i])
                print()
            #load initial S and P sets
            set_dict= initialize_sets(epsilon_percentage)


            set_dict['Vh'] = calculate_vh(gp_mod.kernel_list, rho, alpha, delta, set_dict['points'].shape[0], set_dict['m'],
                                          experiment_setup.design_dim, set_dict['h'], N_refine , v1, kernel_type)

            with h5py.File(save_dir + '/' + 'exp.h5', 'a') as hf:

                hf.create_dataset(
                    'epsilon_percent_{}_iter_{}_eval_{}_Pareto_set_normalized'.format(epsilon_percentage,
                                                                                      iteration, number_of_evaluations),
                    data=set_dict['points'])

            confidence_ratio = experiment_setup.confidence_ratio
            objective_dim = experiment_setup.obj_dim
            design_dim= experiment_setup.design_dim

            refine = False
            evaluate = False
            T = 1

            theo_hmax= 0
            while True:
                vh= calculate_vh(gp_mod.kernel_list, rho, alpha, delta, 1, objective_dim,design_dim , np.array([[theo_hmax]]), N_refine, v1, kernel_type)
                if np.all(16* np.power(vh , 2) * objective_dim <= power(set_dict['epsilon'] , 2)):
                    break
                theo_hmax += 1

            if check_confidence_bounds:
                conf_ratio_list = []
                max_diff_confidence_list = []

            while True:

                if evaluate or T == 1: #since beta only depends on number of evaluations
                    beta= np.zeros([objective_dim,1])
                    for i in range(objective_dim):
                        delta = 1- confidence_ratio
                        if theo_hmax > set_dict['h_max']:
                            beta[i, 0] = 2 * log(2 * set_dict['m'] * power(pi , 2) * power(N_refine, set_dict['h_max'] + 1)  *
                                                             power(number_of_evaluations + 1 , 2) / (3 * delta))
                        else:
                            beta[i, 0] = 2 * log(2 * set_dict['m'] * power(pi , 2) * power(N_refine, theo_hmax + 1)  *
                                                             power(number_of_evaluations + 1 , 2) / (3 * delta))

                """MODELING PHASE"""

                #regression and prediction with gp models
                if not refine or T == 1:
                    if number_of_evaluations>0:
                        gp_mod.update() #update with new evaluations
                        set_dict['mean'], set_dict['std'] = gp_mod.predict(set_dict['points'])

                    else:

                        std_array= np.zeros([1, len(gp_mod.kernel_list)])
                        for i in range(len(gp_mod.kernel_list)):
                            std_array[0, i]= np.sqrt(gp_mod.kernel_list[i].variance[0])

                        set_dict['std']= (np.ones([set_dict['points'].shape[0], set_dict['m']])) * std_array
                        set_dict['mean'] = np.zeros([set_dict['points'].shape[0], set_dict['m']])

                    #update objective unceartainty regions

                    set_dict['objective_cells']= update_objective_cells(set_dict['mean'], set_dict['std'], set_dict['Vh'], beta, flag_hmax)

                    #check confidence bounds
                    if experiment_setup.check_confidence_bounds:
                        _conf_ratio_list, _diff_obs_list = check_beta_confidence_intervals(set_dict['mean'],
                                                                                           set_dict['std'], beta,
                                                                                           objective_dim, design_dim,
                                                                                           objective_function, x_lim,
                                                                                           set_dict['points'])
                        conf_ratio_list.append(_conf_ratio_list)
                        max_diff_confidence_list.append(_diff_obs_list)

                """MODELING PHASE ENDS"""



                # check_beta_confidence_intervals(set_dict['mean'], set_dict['std'], beta, objective_dim, design_dim, objective_function, x_lim, set_dict['points'], print_true = True)
                """DISCARDING PHASE"""
                if number_of_evaluations >= experiment_setup.N_discard:
                    set_dict= discarding(set_dict['objective_cells'], set_dict['points'], set_dict['m'], set_dict['epsilon'], set_dict['nt'], set_dict, flag_hmax, cont=True)

                    #Check if St is empty,terminate  if it is empty
                    if set_dict['nt'].shape[0]==0:
                        print('No point left')
                        break

                    flag_hmax_prev= np.copy(flag_hmax)
                    flag_hmax, set_dict= check_max_depth(flag_hmax, set_dict['h'], h_max, set_dict)

                    if flag_hmax_prev==0 and flag_hmax==1:
                        set_dict['objective_cells'] = \
                            update_objective_cells(set_dict['mean'], set_dict['std'], set_dict['Vh'], beta, flag_hmax)


                """EPSILON-COVERING PHASE """
                _ , max_radius = find_cell_with_maximum_objective_radius(set_dict['objective_cells'], objective_dim, return_max_value = True)
                if np.all(max_radius <= set_dict['epsilon']):
                    set_dict= epsilon_covering(set_dict['points'], set_dict['nt'].shape[0], set_dict['objective_cells'] ,
                                                     experiment_setup.objective_dim , set_dict['epsilon']
                                                       , set_dict, flag_hmax, cont= True)

                # terminate if St is empty
                if set_dict['nt'].shape[0]==0:
                    print('No point left')
                    break

                flag_hmax_prev= np.copy(flag_hmax)
                flag_hmax, set_dict= check_max_depth(flag_hmax, set_dict['h'], h_max, set_dict)
                if flag_hmax_prev==0 and flag_hmax==1:
                    set_dict['objective_cells'] = \
                        update_objective_cells(set_dict['mean'], set_dict['std'], set_dict['Vh'], beta, flag_hmax)


                '''save predicted set after every evaluation'''
                if number_of_evaluations!= number_of_evaluations_prev_write:
                    with h5py.File(save_dir + '/' + 'exp.h5', 'a') as hf:

                        hf.create_dataset(
                            'epsilon_percent_{}_iter_{}_eval_{}_pareto_set_normalized'.format(epsilon_percentage,
                                                      iteration, number_of_evaluations), data=set_dict['points'])

                        hf.create_dataset(
                            'epsilon_percent_{}_iter_{}_eval_{}_pareto_front'.format(epsilon_percentage,
                                                    iteration, number_of_evaluations),data=objective_function(denormalizeX(set_dict['points'], scalex, minx)))

                    number_of_evaluations_prev_write= np.copy(number_of_evaluations)



                """REFINING/EVALUATING PHASE"""

                refine= False
                evaluate= False

                if flag_hmax==0:

                    objective_cells_temp= set_dict['objective_cells'].copy()
                    index_list_temp = np.arange(set_dict['points'].shape[0])

                    while True:

                        index_temp = find_cell_with_maximum_objective_radius(objective_cells_temp, set_dict['m'])
                        index = index_list_temp[index_temp]

                        # if refining condition satisfied and node is at maximum depth


                        if np.linalg.norm(beta ** (1 / 2) * set_dict['std'][index, :].reshape(-1, 1)) < \
                                np.linalg.norm(set_dict['Vh'][index, :].reshape(-1,1)) and set_dict['h'][index, 0]== set_dict['h_max']:

                            index_list_temp= np.delete(index_list_temp, index_temp)
                            objective_cells_temp= np.delete(objective_cells_temp,  index_temp, axis=0);

                        # if only refining condition satisfied
                        elif np.linalg.norm(beta ** (1 / 2) * set_dict['std'][index, :].reshape(-1, 1)) <= \
                                np.linalg.norm(set_dict['Vh'][index, :].reshape(-1,1)):
                            refine = True
                            chosen_index = index
                            break

                        elif np.linalg.norm(beta ** (1 / 2) * set_dict['std'][index, :].reshape(-1, 1)) > \
                                np.linalg.norm(set_dict['Vh'][index, :].reshape(-1,1)):
                            evaluate = True
                            chosen_index = index
                            break


                else:
                    chosen_index = find_cell_with_maximum_objective_radius(set_dict['objective_cells'], set_dict['m'])
                    evaluate= True

                if evaluate:
                    print()
                    print('Tau: ', number_of_evaluations+1)
                    print('----------')
                    if number_of_evaluations>0:
                        set_dict['XT'],set_dict['YT'], xt, yt=evaluate_design(chosen_index, set_dict['points'], set_dict['XT'],
                                                                      set_dict['YT'], scalex, scaley, minx, miny, noise_obj_std)

                    else:
                        set_dict['XT'],set_dict['YT'], xt, yt= evaluate_design(chosen_index, set_dict['points'], set_dict['XT'],
                                                                       set_dict['YT'], scalex, scaley, minx, miny, noise_obj_std)
                    gp_mod.add_sample(xt, yt)

                    print('number of undecided node points: ', set_dict['points'].shape[0])

                if refine:

                    refined_point_index=  chosen_index.copy()

                    #Refining step. New lists are stored inside 'Set' object

                    set_dict= refine_nodes(set_dict, refined_point_index, set_dict['nt'].shape[0] , set_dict['d'] ,
                                           N_refine , gp_mod.kernel_list , number_of_evaluations , rho , alpha , delta , beta , v1, gp_mod, kernel_type)

                    if set_dict['points'].shape[0]> np.inf:
                        flag_excess= True
                        break

                flag_hmax, set_dict= check_max_depth(flag_hmax, set_dict['h'], h_max, set_dict)


                """REFINING - EVALUATING PHASE ENDS """


                if evaluate:
                    number_of_evaluations+= 1
                elif refine:
                    number_of_refinements+= 1
                T+= 1


            if flag_excess:
                break


            """WRITE RESULTS AFTER EACH ITERATION"""
            with h5py.File(save_dir + '/'+  'exp.h5', 'a') as hf:

                hf.create_dataset('epsilon_percent_{}_iter_{}_XT'.format(epsilon_percentage, iteration), data= set_dict['XT'])
                hf.create_dataset('epsilon_percent_{}_iter_{}_YT'.format(epsilon_percentage, iteration), data= set_dict['YT'])
                hf.create_dataset('epsilon_percent_{}_iter_{}_Pareto_set_normalized'.format(epsilon_percentage, iteration), data= set_dict['points'])

                hf.create_dataset('epsilon_percent_{}_iter_{}_Pareto_front_normalized'.format(epsilon_percentage, iteration),
                                  data= objective_function(denormalizeX(set_dict['points'], scalex, minx)))
                # hf.create_dataset('epsilon_percent_{}_iter_{}_Xtr_normalized'.format(epsilon_percentage, iteration), data= Xtrain_n )
                # hf.create_dataset('epsilon_percent_{}_iter_{}_Ytr_normalized'.format(epsilon_percentage, iteration), data= Ytrain_n)

                for k in range(len(gp_mod.kernel_list)):
                    hf.create_dataset('epsilon_percent_{}_iter_{}_kern_{}_ls'.format(epsilon_percentage, iteration, k),
                                      data=(gp_mod.kernel_list[i]).lengthscale.values)
                    hf.create_dataset('epsilon_percent_{}_iter_{}_kern_{}_var'.format(epsilon_percentage, iteration, k),
                                      data=(gp_mod.kernel_list[i]).variance.values)

                    hf.create_dataset('epsilon_percent_{}_iter_{}_noise_{}'.format(epsilon_percentage, iteration, k),
                                      data=(np.array(gp_mod.noise_var_list[i])))

            if check_confidence_bounds:
                pickle_save(os.path.join(save_dir, 'conf_ratio_and_deviation_from_conf'), data= (conf_ratio_list, max_diff_confidence_list))



            print('total evaluations:{}'.format(set_dict['XT'].shape[0]))


            points= set_dict['points'].copy()

            print('predicted pareto set', points)

            samplings[iteration, epsilon_index] = number_of_evaluations
            refinements[iteration, epsilon_index]= number_of_refinements
            print('number of refinements: ', number_of_refinements)
            print('number of evaluations: ', number_of_evaluations)
        if flag_excess:
            print('too many points, terminating')
            break

    return samplings, refinements
