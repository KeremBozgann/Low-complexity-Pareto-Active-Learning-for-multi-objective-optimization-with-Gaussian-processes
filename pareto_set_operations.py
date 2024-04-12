import numpy as np
from loss_calculations import get_pareto_front_set , get_pareto_front_set_special_m2
from confidence_calculation_related_operations import gp_prediction, calculate_vh, update_objective_cells


def find_cell_with_maximum_objective_radius(objective_cells, m ,  return_max_value = False):

    objective_cells_l = objective_cells[:, [2 * i + 1 for i in range(m)]]
    objective_cells_u = objective_cells[:, [2 * i for i in range(m)]]

    if return_max_value:
        arg_max_rad = np.argmax(np.linalg.norm(objective_cells_u - objective_cells_l , axis = 1)[:])
        return  arg_max_rad , (objective_cells_u[arg_max_rad] - objective_cells_l[arg_max_rad]).reshape(1 , -1)
    else:
        return np.argmax(np.linalg.norm(objective_cells_u - objective_cells_l , axis = 1)[:])


def get_index_maximum_objective_radius(objective_cells, m):
    objective_cells_l= objective_cells[:, [2*i+1 for i in range(m)]]
    objective_cells_u= objective_cells[:, [2*i for i in range(m)]]

    square_dist= np.zeros([objective_cells.shape[0], 1])

    for i in range(m):
        dist_i = (objective_cells_u[:, i].reshape(-1, 1) - objective_cells_l[:, i].reshape(-1, 1)) ** 2
        square_dist = square_dist + dist_i

    dist = np.sqrt(square_dist)

    max_index = np.argmax(dist)

    return max_index



def update_dict(set_dict, keep_index_bool, nt_length, flag_hmax, cont):
    keep_index= np.arange(nt_length)[keep_index_bool]
    nt_length_new = np.sum(keep_index_bool)
    np_length = set_dict['points'].shape[0] - nt_length
    tot_num_points = nt_length_new + np_length
    d = set_dict['points'].shape[1]
    m = set_dict['mean'].shape[1]

    if flag_hmax==0:

        set_dict['nt']= set_dict['nt'][keep_index,:]

        mean_new = np.empty([tot_num_points, m])
        mean_new[:nt_length_new] = set_dict['mean'][0:nt_length, :][keep_index, :]
        mean_new[nt_length_new:] = set_dict['mean'][nt_length:, :]
        set_dict['mean'] = mean_new

        std_new = np.empty([tot_num_points, m])
        std_new[:nt_length_new] = set_dict['std'][0:nt_length, :][keep_index, :]
        std_new[nt_length_new:] = set_dict['std'][nt_length:, :]
        set_dict['std'] = std_new

        objective_cells_new = np.empty([tot_num_points, 2 * m])
        objective_cells_new[:nt_length_new] = set_dict['objective_cells'][0:nt_length, :][keep_index, :]
        objective_cells_new[nt_length_new:] = set_dict['objective_cells'][nt_length:, :]
        set_dict['objective_cells'] = objective_cells_new

        design_cells_new = np.empty([tot_num_points, 2 * d])
        design_cells_new[:nt_length_new] = set_dict['design_cells'][0:nt_length, :][keep_index, :]
        design_cells_new[nt_length_new:] = set_dict['design_cells'][nt_length:, :]
        set_dict['design_cells'] = design_cells_new

        h_new = np.empty([tot_num_points, 1])
        h_new[:nt_length_new] = set_dict['h'][0:nt_length, :][keep_index, :]
        h_new[nt_length_new:] = set_dict['h'][nt_length:, :]
        set_dict['h'] = h_new

        points_new= np.empty([tot_num_points, d])
        points_new[:nt_length_new] = set_dict['points'][0:nt_length, :][keep_index, :]
        points_new[nt_length_new:] = set_dict['points'][nt_length:, :]
        set_dict['points'] = points_new

        Vh_new = np.empty([tot_num_points, m])
        Vh_new[:nt_length_new] = set_dict['Vh'][0:nt_length, :][keep_index, :]
        Vh_new[nt_length_new:] = set_dict['Vh'][nt_length:, :]
        set_dict['Vh'] = Vh_new

    else:

        mean_new = np.empty([tot_num_points, m])
        mean_new[:nt_length_new] = set_dict['mean'][0:nt_length, :][keep_index, :]
        mean_new[nt_length_new:] = set_dict['mean'][nt_length:, :]
        set_dict['mean'] = mean_new

        std_new = np.empty([tot_num_points, m])
        std_new[:nt_length_new] = set_dict['std'][0:nt_length, :][keep_index, :]
        std_new[nt_length_new:] = set_dict['std'][nt_length:, :]
        set_dict['std'] = std_new

        objective_cells_new = np.empty([tot_num_points, 2 * m])
        objective_cells_new[:nt_length_new] = set_dict['objective_cells'][0:nt_length, :][keep_index, :]
        objective_cells_new[nt_length_new:] = set_dict['objective_cells'][nt_length:, :]
        set_dict['objective_cells'] = objective_cells_new

        points_new= np.empty([tot_num_points, d])
        points_new[:nt_length_new] = set_dict['points'][0:nt_length, :][keep_index, :]
        points_new[nt_length_new:] = set_dict['points'][nt_length:, :]
        set_dict['points'] = points_new

        set_dict['nt']= set_dict['nt'][keep_index,:]

        if not cont:
            dataY_new = np.empty([tot_num_points, m])
            dataY_new[:nt_length_new] = set_dict['dataY'][0:nt_length, :][keep_index, :]
            dataY_new[nt_length_new:] = set_dict['dataY'][nt_length:, :]
            set_dict['dataY'] = dataY_new

    return set_dict


def update_dict_epal(set_dict, keep_index_bool, nt_length):

    keep_index= np.arange(nt_length)[keep_index_bool]
    nt_length_new = np.sum(keep_index_bool)
    np_length = set_dict['points'].shape[0] - nt_length
    tot_num_points = nt_length_new + np_length
    d = set_dict['points'].shape[1]
    m = set_dict['points_obj'].shape[1]

    points_new = np.empty([tot_num_points, d])
    points_new[:nt_length_new] = set_dict['nt'][keep_index,:]
    points_new[nt_length_new:] =set_dict['points'][nt_length:, :]
    set_dict['points'] = points_new

    mean_new = np.empty([tot_num_points, m])
    mean_new[:nt_length_new] = set_dict['mean'][0:nt_length, :][keep_index, :]
    mean_new[nt_length_new:] = set_dict['mean'][nt_length:,:]
    set_dict['mean'] = mean_new
    # set_dict['mean']= np.append(set_dict['mean'][0:nt_length, :][keep_index, :], set_dict['mean'][nt_length:,:], axis=0)

    std_new = np.empty([tot_num_points, m])
    std_new [:nt_length_new] = set_dict['std'][0:nt_length, :][keep_index, :]
    std_new [nt_length_new:] = set_dict['std'][nt_length:,:]
    set_dict['std'] = std_new

    objective_cells_new = np.empty([tot_num_points, 2*m])
    objective_cells_new[:nt_length_new] = set_dict['objective_cells'][0:nt_length, :][keep_index, :]
    objective_cells_new[nt_length_new:] = set_dict['objective_cells'][nt_length:, :]
    set_dict['objective_cells'] = objective_cells_new

    set_dict['nt'] = set_dict['nt'][keep_index, :]

    points_obj_new = np.empty([tot_num_points, m])
    points_obj_new[:nt_length_new] = set_dict['points_obj'][0:nt_length, :][keep_index,:]
    points_obj_new[nt_length_new:] = set_dict['points_obj'][nt_length:, :]
    set_dict['points_obj'] = points_obj_new

    return set_dict

def get_pessimistic_pareto(points, objective_cells_l, m , return_pess_front=False):

    if m==2:
        pessimistic_pareto_set, pessimistic_pareto_front, pess_indeces = get_pareto_front_set_special_m2(points, objective_cells_l, get_indeces=True)
    else:
        pessimistic_pareto_set, pessimistic_pareto_front, pess_indeces= get_pareto_front_set(points, objective_cells_l, get_indeces=True)

    if return_pess_front:
        return pessimistic_pareto_set, pessimistic_pareto_front ,pess_indeces
    else:
        return pessimistic_pareto_set,pess_indeces


def test_get_pessimistic_pareto():
    m=2
    d=2
    d=2
    points= np.random.normal(0.0, 1.0, (10,2))
    objective_cells= np.random.normal(0.0, 1.0, (10,4))
    print('points')
    print(points)
    print('objective_cells')
    print(objective_cells)
    pessimistic_set, pessimistic_index= get_pessimistic_pareto(points, objective_cells)
    print('pessimistic set')
    print(pessimistic_set)
    print('pessimistic index')
    print(pessimistic_index)

def discarding(objective_cells, points, m, epsilon, n_t , set_dict, flag_hmax, cont):
    objective_cells_l = objective_cells[:, [2 * i + 1 for i in range(m)]]
    objective_cells_u = objective_cells[:, [2 * i  for i in range(m)]]

    pessimistic_pareto_set, pessimistic_pareto_front, pessimistic_pareto_set_index = get_pessimistic_pareto(points,
                                                                                                            objective_cells_l, m ,
                                                                                                            return_pess_front=True)

    if m == 2:
        n_s = n_t.shape[0]
        objective_cells_u_s = objective_cells_u[:n_s]
        n_tot = objective_cells.shape[0]

        keep_index = special_algorithm_for_discarding_m2(pessimistic_pareto_front, objective_cells_u_s, epsilon,
                                                         pessimistic_pareto_set_index , n_s , n_tot , use_for_loop = False)


    else:
        pessimistic_pareto_front= objective_cells_l[ pessimistic_pareto_set_index, :]
        keep_index= np.ones([n_t.shape[0], 1])

        for i in range(n_t.shape[0]):

           vi= objective_cells_u[i, :].reshape(-1,1)

           for j in range(pessimistic_pareto_front.shape[0]):
               vj= pessimistic_pareto_front[j, :].reshape(-1,1)

               if not i in  pessimistic_pareto_set_index:
                   if np.all(vj+ epsilon >= vi):
                       keep_index[i, 0]= 0
                       break
        keep_index = keep_index.astype(bool)

    set_dict= update_dict(set_dict, keep_index, n_t.shape[0], flag_hmax, cont)

    return  set_dict

#TODO write a test for discarding
def test_discarding():
    pass

def special_algorithm_for_discarding_m2_unoptimized(pessimistic_pareto_front, objective_cells_u, epsilon,
                                                pessimistic_pareto_set_index):
    except_pess_index = np.setdiff1d(np.arange(objective_cells_u.shape[0]), pessimistic_pareto_set_index)
    objective_cells_u_except_pess = objective_cells_u[except_pess_index]

    pessimistic_pareto_front_eps_added= pessimistic_pareto_front+epsilon
    n_pess = pessimistic_pareto_front_eps_added.shape[0]
    # TODO check the speed of np.append vs. defining S_tild beforehand with zeros array and assigning values to its elements

    S_tild= np.append(pessimistic_pareto_front_eps_added, objective_cells_u_except_pess, axis=0)
    index_map1 = np.append(pessimistic_pareto_set_index, except_pess_index) #map altered array elements to their original order in the source array

    index_map_sort = np.argsort(-S_tild[:, 0]) #map sorted array elements to their original order in the source array

    S_sorted= S_tild[index_map_sort]

    n_for = S_sorted.shape[0]

    kept_indeces = np.ones([n_for], dtype = bool)

    current_max = -np.inf
    for i in range(n_for):
        r= S_sorted[i, :]
        if index_map1[index_map_sort[i]] in pessimistic_pareto_set_index: #if array belongs to pessimistic set
            current_max= r[1]
        else:
            if r[1]<= current_max:
                kept_indeces[index_map1[index_map_sort[i]]]= False

    return kept_indeces

def special_algorithm_for_discarding_m2(pessimistic_pareto_front, objective_cells_u_s, epsilon,
                                                pessimistic_pareto_set_index , n_s , n_tot , use_for_loop = False):


    if use_for_loop:
        pess_s_bool = pessimistic_pareto_set_index < n_s
        except_pess_index = np.setdiff1d(np.arange(objective_cells_u_s.shape[0]) , pessimistic_pareto_set_index[pess_s_bool])
        objective_cells_u_except_pess = objective_cells_u_s[except_pess_index]

        pessimistic_pareto_front_eps_added= pessimistic_pareto_front+epsilon
        n_pess = pessimistic_pareto_front_eps_added.shape[0]
        # TODO check the speed of np.append vs. defining S_tild beforehand with zeros array and assigning values to its elements

        S_tild= np.append(pessimistic_pareto_front_eps_added, objective_cells_u_except_pess, axis=0)
        index_map1 = np.append(pessimistic_pareto_set_index, except_pess_index) #map altered array elements to their original order in the source array

        index_map_sort = np.argsort(-S_tild[:, 0]) #map sorted array elements to their original order in the source array

        S_sorted= S_tild[index_map_sort]


        temp_sort_bool_object = index_map_sort <= (n_pess-1)
        pess_sorting_index =  index_map_sort[temp_sort_bool_object]
        pessimistic_pareto_front_eps_added_sorted =  pessimistic_pareto_front_eps_added[pess_sorting_index]
        pess_index_ordering_inside_sorted_array = temp_sort_bool_object.nonzero()[0]

        n_for = S_sorted.shape[0]

        kept_indeces_true_false_sorted_array = np.ones([n_for], dtype = bool)

        kept_indeces_true_false = np.ones([n_tot], dtype = bool) #including pareto set P

        # TODO implementation of indeces kept with indeces rather than boolean
        # TODO optimize this code even more
        for i in range(n_pess):
            v_pes_obj2 = pessimistic_pareto_front_eps_added_sorted[i][1]
            order_in_sorted_array = pess_index_ordering_inside_sorted_array[i]

            if order_in_sorted_array == (n_for-1):
                break
            elif i == (n_pess-1):
                kept_indeces_true_false_sorted_array[order_in_sorted_array+1:] = (S_sorted[order_in_sorted_array+1:][:, 1] > v_pes_obj2)
            else:
                order_in_sorted_array_next_pess_element = pess_index_ordering_inside_sorted_array[i+1]
                kept_indeces_true_false_sorted_array[order_in_sorted_array+1:order_in_sorted_array_next_pess_element] =\
                    (S_sorted[order_in_sorted_array+1:order_in_sorted_array_next_pess_element][:, 1] > v_pes_obj2)

        kept_indeces_true_false[index_map1[index_map_sort]] = kept_indeces_true_false_sorted_array

    else:
        n_pess = len(pessimistic_pareto_set_index)

        except_pess_bool = np.ones([n_s] , dtype = bool)
        pess_s_bool = pessimistic_pareto_set_index < n_s

        except_pess_bool[pessimistic_pareto_set_index[pess_s_bool]] = False
        except_pess_index = np.arange(n_s)[except_pess_bool]

        pessimistic_pareto_front_eps_added= pessimistic_pareto_front+epsilon
        objective_cells_u_s_except_pess = objective_cells_u_s[except_pess_index]

        S_tild = np.append(pessimistic_pareto_front_eps_added, objective_cells_u_s_except_pess, axis=0)
        index_map1 = np.append(pessimistic_pareto_set_index, except_pess_index) #map altered array elements to their original order in the source array

        index_map_sort = np.argsort(-S_tild[:, 0]) #map sorted array elements to their original order in the source array
        S_sorted= S_tild[index_map_sort]

        n_for = S_sorted.shape[0]
        obj2 = S_sorted[:, 1]

        kept_indeces_true_false = np.ones(n_tot, dtype=bool)

        #remove effect of upper confidence bound cells
        _obj2 = obj2.copy()
        _obj2[index_map_sort>=n_pess] = -np.inf

        accumulated_max = np.zeros([n_for])
        accumulated_max[0] = -np.inf

        accumulated_max[1:] = (np.maximum.accumulate(_obj2))[:-1]

        kept_indeces_sorted_true_false = obj2 > accumulated_max

        # example
        # kept_indeces_true_false[index_map1[index_map_sort][0]] kept_indeces_sorted_true_false[0]
        kept_indeces_true_false[index_map1[index_map_sort]]= kept_indeces_sorted_true_false

    return kept_indeces_true_false[:n_s]


def test_special_algorithm():
    import time


    n_pes = 1000
    n_tot = 1048576

    count1 = 0
    count2 = 0
    count3 = 0
    for i in range(100):
        obj1 = np.sort(np.random.rand(n_pes, 1), axis=0)[::-1]
        obj2 = np.sort(np.random.rand(n_pes, 1) , axis = 0)
        pessimistic_pareto_front= np.append(obj1 , obj2 , axis=1)

        pessimistic_pareto_set_index= np.random.permutation(n_pes)
        objective_cells_u = np.random.rand(n_tot,2)
        epsilon = 0.01

        start1 = time.time()
        kept_indeces1= special_algorithm_for_discarding_m2(pessimistic_pareto_front, objective_cells_u, epsilon,
                                            pessimistic_pareto_set_index, use_for_loop = False)
        stop1 = time.time()

        start2 = time.time()
        kept_indeces2= special_algorithm_for_discarding_m2(pessimistic_pareto_front, objective_cells_u, epsilon,
                                            pessimistic_pareto_set_index, use_for_loop = True)
        stop2 = time.time()

        start3 = time.time()
        kept_indeces3= special_algorithm_for_discarding_m2_unoptimized(pessimistic_pareto_front, objective_cells_u, epsilon,
                                            pessimistic_pareto_set_index)
        stop3 = time.time()
        assert np.all(kept_indeces1==kept_indeces2) and np.all(kept_indeces2 == kept_indeces3)
        print()
        print()
        print(f'validate methods by comparing returned indeces :{np.all(kept_indeces1==kept_indeces2) and np.all(kept_indeces2 == kept_indeces3)}')
        print(f'special m2 without for loop run-time :{stop1-start1}')
        print(f'special m2 with for loop run-time :{stop2-start2}')
        print(f'non-special run-time :{stop3-start3}')

        count1 += stop1 - start1 < stop2 - start2 and stop1 - start1 < stop3 - start3
        count2 += stop2 - start2 < stop1 - start1 and stop2 - start2 < stop3 - start3
        count3 += stop3 - start3 < stop2 - start2 and stop3 - start3 < stop1 - start1
    print(f'number of without for loop wins : {count1}')
    print(f'number of with for loop wins : {count2}')
    print(f'number of non-special wins : {count3}')


def epsilon_covering(points, nt_length, objective_cells,  m , epsilon,  set_dict, flag_hmax, cont , force_for_loop_for_test = False):

    d = points.shape[1]
    tot_num_points = objective_cells.shape[0]

    objective_cells_nt= set_dict['objective_cells'][0:nt_length]
    objective_cells_pt= set_dict['objective_cells'][nt_length:]

    nt = set_dict['points'][0:nt_length, :]
    pt = set_dict['points'][nt_length:, :]

    if flag_hmax==1 and not cont:
        dataY_nt= set_dict['dataY'][0:nt_length, :]
        dataY_pt= set_dict['dataY'][nt_length:, :]

    else:
        h_nt= set_dict['h'][ 0: nt_length, :]
        h_pt= set_dict['h'][nt_length:, :]

        Vh_nt= set_dict['Vh'][ 0: nt_length, :]
        Vh_pt= set_dict['Vh'][nt_length:, :]

        design_cells_nt= set_dict['design_cells'][0:nt_length, :]
        design_cells_pt= set_dict['design_cells'][nt_length:, :]


    objective_cells_nt_l = objective_cells[0:nt_length][:, [2 * i + 1 for i in range(m)]]
    objective_cells_nt_u = objective_cells[0:nt_length][:, [2 * i  for i in range(m)]]

    objective_cells_pt_u= objective_cells[nt_length:][:, [2*i for i in range(m)]]

    objective_cells_u= np.append(objective_cells_nt_u, objective_cells_pt_u, axis= 0)

    if m==2 and not force_for_loop_for_test:
        choose_indeces = special_epsilon_covering_algorithm_m2(points, objective_cells_nt_l,  objective_cells_u,
                                          nt_length, epsilon, m , use_for_loop = False)
        keep_indeces_nt_bool = np.ones(nt_length , dtype = bool)
        keep_indeces_nt_bool[choose_indeces] = False
        keep_indeces_nt = np.arange(nt_length)[keep_indeces_nt_bool]

    else:
        choose= np.ones([nt_length, 1])
        keep_nt= np.zeros([nt_length, 1])

        for i in range(nt_length):
            vi= objective_cells_nt_l[i, :].reshape(-1, 1)

            for j in range(points.shape[0]):
                vj= objective_cells_u[j, :].reshape(-1,1)

                if np.all(vj>= vi+epsilon):
                    choose[i, 0] = 0
                    keep_nt[i, 0]= 1

        keep_indeces_nt= np.where(keep_nt==1)[0]
        choose_indeces= np.where(choose==1)[0]

    num_s_new = len(keep_indeces_nt)
    objective_cells_nt_new= objective_cells_nt[keep_indeces_nt, :]
    nt_new= nt[keep_indeces_nt, :]

    num_p_prev = objective_cells_pt.shape[0]
    num_p_new = len(choose_indeces)
    objective_cells_pt_new = np.empty([num_p_prev + num_p_new  , 2*m])
    objective_cells_pt_new[:num_p_prev , :] = objective_cells_pt
    objective_cells_pt_new[num_p_prev:, :] = objective_cells_nt[choose_indeces, :]

    pt_new = np.empty([num_p_prev + num_p_new  , d])
    pt_new[:num_p_prev , :] = pt
    pt_new[num_p_prev: , :] = nt[choose_indeces, :]


    if flag_hmax==1 and not cont:
        dataY_nt_new= dataY_nt[keep_indeces_nt, :]
        dataY_pt_new = np.append(dataY_pt, dataY_nt[choose_indeces, :], axis=0)

    elif flag_hmax==0:
        design_cells_nt_new= design_cells_nt[keep_indeces_nt, :]
        h_nt_new= h_nt[keep_indeces_nt, :]
        Vh_nt_new= Vh_nt[keep_indeces_nt, :]

        design_cells_pt_new = np.empty([num_p_prev + num_p_new, 2 * d])
        design_cells_pt_new[:num_p_prev, :] = design_cells_pt
        design_cells_pt_new[num_p_prev:, :] = design_cells_nt[choose_indeces, :]

        # design_cells_pt_new= np.append(design_cells_pt,design_cells_nt[choose_indeces, :] , axis=0)
        h_pt_new = np.empty([num_p_prev + num_p_new, 1])
        h_pt_new[:num_p_prev, :] = h_pt
        h_pt_new[num_p_prev:, :] = h_nt[choose_indeces, :]

        # h_pt_new= np.append(h_pt, h_nt[choose_indeces, :] , axis=0)
        Vh_pt_new = np.empty([num_p_prev + num_p_new, m])
        Vh_pt_new[:num_p_prev, :] = Vh_pt
        Vh_pt_new[num_p_prev:, :] = Vh_nt[choose_indeces, :]

    set_dict['nt']= nt_new

    set_dict['points'] = np.empty([tot_num_points, d])
    set_dict['points'][:num_s_new] = nt_new
    set_dict['points'][num_s_new:] = pt_new

    set_dict['objective_cells'] = np.empty([tot_num_points , 2*m])
    set_dict['objective_cells'][:num_s_new] = objective_cells_nt_new
    set_dict['objective_cells'][num_s_new:] =  objective_cells_pt_new

    if flag_hmax==1 and not cont:
        set_dict['dataY'] = np.empty([tot_num_points, m])
        set_dict['dataY'][:num_s_new] = dataY_nt_new
        set_dict['dataY'][num_s_new:] = dataY_pt_new

    elif flag_hmax==0:
        set_dict['h'] = np.empty([tot_num_points, 1])
        set_dict['h'][:num_s_new] = h_nt_new
        set_dict['h'][num_s_new:] = h_pt_new

        set_dict['Vh'] = np.empty([tot_num_points, m])
        set_dict['Vh'][:num_s_new] = Vh_nt_new
        set_dict['Vh'][num_s_new:] = Vh_pt_new

        set_dict['design_cells'] = np.empty([tot_num_points, 2*d])
        set_dict['design_cells'][:num_s_new] = design_cells_nt_new
        set_dict['design_cells'][num_s_new:] = design_cells_pt_new

    return set_dict


def test_epsilon_covering():

    set_dict= dict()
    set_dict['nt']= np.random.normal(0.0, 1.0, (10,2))
    set_dict['pt']= np.random.normal(0.0, 1.0, (5,2))
    set_dict['points']= np.append(set_dict['nt'], set_dict['pt'], axis=0)


    d= 2
    set_dict['design_cells_nt'] = np.ones([10, 4])
    for i in range(2):
        set_dict['design_cells_nt'][:, 2*i+1]= np.zeros([10, ])

    set_dict['design_cells_pt'] = np.ones([5, 4])
    for i in range(2):
        set_dict['design_cells_pt'][:, 2*i+1]= np.zeros([5,])

    set_dict['design_cells']= np.append(set_dict['design_cells_nt'], set_dict['design_cells_pt'], axis= 0)

    m=2
    set_dict['objective_cells_nt'] = np.ones([10, 4])
    for i in range(2):
        set_dict['objective_cells_nt'][:, 2*i+1]= np.zeros([10, ])

    set_dict['objective_cells_pt'] = np.ones([5, 4])
    for i in range(2):
        set_dict['objective_cells_pt'][:, 2*i+1]= np.zeros([5,])

    set_dict['objective_cells']= np.append(set_dict['objective_cells_nt'], set_dict['objective_cells_pt'], axis= 0)

    set_dict['Vh']= np.append(np.arange(10, 0, -1).reshape(-1, 1), np.arange(10, 0, -1).reshape(-1, 1), axis= 1)
    set_dict['h']= np.arange(10, 0, -1).reshape(-1, 1)

    nt_length= set_dict['nt'].shape[0]
    points= set_dict['points'].copy()
    epsilon= 5
    set_dict= epsilon_covering(points, nt_length, set_dict['objective_cells_nt'], set_dict['objective_cells_pt'], m, epsilon, set_dict )
    print(set_dict)


def unoptimized_epsilon_covering(points , objective_cells_nt_l , objective_cells_u , nt_length , epsilon):

    choose = np.ones([nt_length, 1])
    keep_nt = np.zeros([nt_length, 1])

    for i in range(nt_length):
        vi = objective_cells_nt_l[i, :].reshape(-1, 1)

        for j in range(points.shape[0]):
            vj = objective_cells_u[j, :].reshape(-1, 1)

            if np.all(vj >= vi + epsilon):
                choose[i, 0] = 0
                keep_nt[i, 0] = 1

    keep_indeces_nt = np.where(keep_nt == 1)[0]
    choose_indeces = np.where(choose == 1)[0]

    return np.arange(nt_length)[choose_indeces]

def _special_epsilon_covering_algorithm(objective_cells_u , lower_confidence_pessimistic , epsilon , use_for_loop = False):
    # except_pess_index = np.setdiff1d(np.arange(objective_cells_u.shape[0]), pessimistic_indeces)
    # objective_cells_u_except_pess = objective_cells_u[except_pess_index]

    lower_confidence_pessimistic_eps_added= lower_confidence_pessimistic+epsilon
    n_pess = lower_confidence_pessimistic_eps_added.shape[0]

    S_tild= np.append(lower_confidence_pessimistic_eps_added, objective_cells_u, axis=0)

    index_map_sort = np.argsort(-S_tild[:, 0]) #map sorted array elements to their original order in the source array

    S_sorted= S_tild[index_map_sort]

    n_for = S_sorted.shape[0]
    obj2 = S_sorted[:, 1]
    if not use_for_loop:
        temp_sort_bool_object = index_map_sort <= (n_pess - 1)
        pess_sorting_index = index_map_sort[temp_sort_bool_object]
        pess_index_ordering_inside_sorted_array = temp_sort_bool_object.nonzero()[0]

        kept_indeces_true_false = np.ones(n_for, dtype=bool)

        accumulated_max = np.zeros([pess_index_ordering_inside_sorted_array[-1]+1])
        accumulated_max[0] = -np.inf

        _obj2 = obj2.copy()
        # remove effect of pessimistic points on accumulated max
        _obj2[pess_index_ordering_inside_sorted_array] = -np.inf

        last_pess_index_in_sorted_array = pess_index_ordering_inside_sorted_array[-1]
        accumulated_max[1:] = (np.maximum.accumulate(_obj2[:last_pess_index_in_sorted_array]))

        kept_indeces_sorted_true_false = obj2[:last_pess_index_in_sorted_array+1] > accumulated_max
        kept_indeces_true_false[index_map_sort[:last_pess_index_in_sorted_array+1]]= kept_indeces_sorted_true_false
        return kept_indeces_true_false[:n_pess]

    else:

        # obj2_epsilon_accurate_pess = obj2[index_map_sort]
        temp_sort_bool_object = index_map_sort <= (n_pess - 1)
        pess_sorting_index = index_map_sort[temp_sort_bool_object]
        lower_confidence_pessimistic_eps_added_sorted = lower_confidence_pessimistic_eps_added[pess_sorting_index]
        pess_index_ordering_inside_sorted_array = temp_sort_bool_object.nonzero()[0]
        kept_indeces_pes = np.ones([n_pess] , dtype = bool)
        obj_pess2 = lower_confidence_pessimistic_eps_added_sorted[:, 1]
        for i in range(n_pess):
            v_pes_obj2 = obj_pess2[i]
            order_in_sorted_array = pess_index_ordering_inside_sorted_array[i]
            if not order_in_sorted_array == 0:

                non_pess_elements_true_false = np.ones(order_in_sorted_array, dtype = bool)
                non_pess_elements_true_false[pess_index_ordering_inside_sorted_array[:i]] = False

                obj2_to_compare_with_pess_element = obj2[:order_in_sorted_array][non_pess_elements_true_false]
                if len(obj2_to_compare_with_pess_element) > 0:
                    if  np.any(np.max(obj2_to_compare_with_pess_element)>= v_pes_obj2):
                        kept_indeces_pes[pess_sorting_index[i]] = False
        return kept_indeces_pes


def special_epsilon_covering_algorithm_m2(points , objective_cells_nt_l , objective_cells_u , nt_length , epsilon , m , use_for_loop = False):

    #find pessimistic of lower bounds
    pessimistic_pareto_set , pessimistic_pareto_front , pess_indeces = get_pessimistic_pareto(points[:nt_length], objective_cells_nt_l , m, return_pess_front= True) #O(2n)

    #find epsilon accurate pessimistic of lower bounds
    kept_indeces_epsilon_accurate_pess_true_false = \
        special_algorithm_for_discarding_m2(pessimistic_pareto_front , objective_cells_nt_l + epsilon , epsilon = 0 ,
                                            pessimistic_pareto_set_index = pess_indeces , n_s= nt_length , n_tot = nt_length , use_for_loop = False) #O(2n)

    index_map_epsilon_accurate_pess = np.arange(nt_length)[kept_indeces_epsilon_accurate_pess_true_false]
    objective_cells_nt_l_left = objective_cells_nt_l[kept_indeces_epsilon_accurate_pess_true_false]

    pareto_chosen_indeces_epsilon_accurate_pess = _special_epsilon_covering_algorithm(objective_cells_u, objective_cells_nt_l_left, epsilon, use_for_loop = use_for_loop)
    pareto_chosen_indeces = index_map_epsilon_accurate_pess[pareto_chosen_indeces_epsilon_accurate_pess]
    return pareto_chosen_indeces


def test_special_epsilon_covering_algorithm_m2():
    nt_length = 100
    p_length = 5
    points =  np.random.rand(nt_length + p_length ,2)
    import time
    objective_cells_nt_l = np.random.rand(nt_length,2)
    # objective_cells_nt_u = np.random.rand(nt_length,2)
    objective_cells_nt_u = objective_cells_nt_l + np.abs(np.random.rand(nt_length , 2))/1000

    objective_cells_pt_u = np.random.rand(p_length,2)
    objective_cells_u = np.append(objective_cells_nt_u , objective_cells_pt_u , axis = 0)
    nt_length = objective_cells_nt_l.shape[0]
    epsilon = 0.01

    m = 2
    # start1 = time.time()
    # index1 = special_epsilon_covering_algorithm_m2(points, objective_cells_nt_l, objective_cells_u,
    #                                       nt_length, epsilon, m , use_for_loop = True)
    # stop1 = time.time()

    start2 = time.time()
    index2 = special_epsilon_covering_algorithm_m2(points, objective_cells_nt_l,  objective_cells_u,
                                          nt_length, epsilon, m , use_for_loop = False)
    stop2 = time.time()


    # start3 = time.time()
    # index3 = unoptimized_epsilon_covering(points, objective_cells_nt_l, objective_cells_u, nt_length,
    #                              epsilon)
    # stop3 = time.time()

    # print(f'special algorithm with for loop run-time {stop1 - start1}')
    print(f'special algorithm without for loop run-time {stop2 - start2}')
    # print(f'unoptimized algorithm run-time {stop3 - start3}')
    assert np.all(index2 == index1)
    print(f'results are eqaul: {np.all(index2 == index1)}')



def refine_nodes(set_dict, refined_point_index, nt_length, d, N_refine , kernels , number_of_evaluations,
                rho , alpha , delta , beta , v1, gp_mod, kernel_type):


    mean_nt= set_dict['mean'][0:nt_length, :]
    mean_pt= set_dict['mean'][nt_length:, :]

    std_nt= set_dict['std'][0:nt_length, :]
    std_pt= set_dict['std'][nt_length:, :]

    Vh_nt= set_dict['Vh'][0:nt_length, :]
    Vh_pt= set_dict['Vh'][nt_length:, :]

    objective_cells_nt= set_dict['objective_cells'][0:nt_length, :]
    objective_cells_pt= set_dict['objective_cells'][nt_length:, :]

    design_cells_nt= set_dict['design_cells'][0:nt_length, :]
    design_cells_pt= set_dict['design_cells'][nt_length:, :]



    h_nt = set_dict['h'][0:nt_length, :]
    h_pt = set_dict['h'][nt_length:, :]
    nt = set_dict['points'][0:nt_length, :]
    pt = set_dict['points'][nt_length:, :]

    pt_length = pt.shape[0]

    design_cells_nt_l = design_cells_nt[:, [2*i+1 for i in range(d)]]
    design_cells_nt_u = design_cells_nt[:, [2*i for i in range(d)]]
    design_cells_pt_l = design_cells_pt[:, [2*i+1 for i in range(d)]]
    design_cells_pt_u = design_cells_pt[:, [2*i for i in range(d)]]

    num_points_init = nt_length + design_cells_pt.shape[0]
    obj_dim = mean_nt.shape[1]


    if refined_point_index<nt_length:
        diff_design= design_cells_nt_u[refined_point_index, :].reshape(1, -1)- design_cells_nt_l[refined_point_index, :].reshape(1, -1)
        h_child= np.ones([N_refine**d, 1])* (h_nt[refined_point_index, 0]+1)
        flag_n= True
    else:
        p_index= refined_point_index- nt_length
        diff_design = design_cells_pt_u[p_index, :].reshape(1, -1)- design_cells_pt_l[p_index, :].reshape(1, -1)
        h_child = np.ones([N_refine**d, 1]) * (h_pt[p_index, 0] + 1)
        flag_n= False

    # refined_dimension= np.argmax(diff_design, axis= 1)[0]

    #get child node center coordinates
    if flag_n:
        coeff = 2* np.arange(N_refine) + 1
        L = design_cells_nt_l[refined_point_index, :]
        diff = diff_design[0 , :]
        grid_list = [L[i]+coeff*(diff[i]/(2*N_refine)) for i in range(d)]
    else:
        coeff = 2* np.arange(N_refine) + 1
        L = design_cells_nt_l[p_index, :]
        diff = diff_design[0 , :]
        grid_list = [L[i]+coeff*(diff[i]/(2*N_refine)) for i in range(d)]


    temp= np.meshgrid(*grid_list)

    child_nodes = np.zeros([N_refine ** d, d])
    design_cells_child = np.zeros([N_refine ** d, 2 * d])

    #get refined design cells and cell upper and lower bounds of child nodes

    for i in range(d):
        child_nodes[:, i] = temp[i].flatten()

        diff = diff_design[0, i]
        delta = diff / (2 * N_refine)
        upper_i = (child_nodes[:, i] + delta).reshape(-1, 1)
        lower_i = (child_nodes[:, i] - delta).reshape(-1, 1)
        design_cells_child[:, 2 * i] = upper_i[:, 0]
        design_cells_child[:, 2 * i + 1] = lower_i[:, 0]

    del temp


    #calculate child node mean and std
    if number_of_evaluations>0:
        # mean_child, std_child = gp_prediction(kernels, child_nodes, set_dict['XT'],
        #                                                   set_dict['YT'], noises)
        mean_child, std_child= gp_mod.predict(child_nodes)
    else:
        std_array = np.zeros([1, len(kernels)])
        for i in range(len(kernels)):
            std_array[0, i] = np.sqrt(kernels[i].variance[0])

        std_child = (np.ones([child_nodes.shape[0], obj_dim])) * std_array
        mean_child = np.zeros([child_nodes.shape[0], obj_dim])

    #calculate vh for child nodes
    Vh_child= calculate_vh(kernels, rho, alpha, delta, N_refine ** d, obj_dim,
                                  d, h_child, N_refine , v1, kernel_type)

    #calculate objective cells for child nodes (since beta does not depend on number of refinings, we dont have to recalculate for other points
    objective_cells_child = \
        update_objective_cells(mean_child, std_child, Vh_child, beta, flag_hmax = False)


    num_points_new = num_points_init - 1 + N_refine ** d

    h_new = np.empty([num_points_new , 1])
    design_cells_new = np.empty([num_points_new , 2 * d])
    mean_new = np.empty([num_points_new , obj_dim])
    std_new = np.empty([num_points_new , obj_dim])
    Vh_new = np.empty([num_points_new , obj_dim])
    objective_cells_new = np.empty([num_points_new , 2 * obj_dim])
    points_new = np.empty([num_points_new , d])

    if flag_n:
        num_points_new_nt = nt_length - 1 + N_refine ** d
        h_nt_new= np.delete(h_nt, refined_point_index, axis= 0)
        h_new[:num_points_new_nt - N_refine ** d] = h_nt_new
        h_new[num_points_new_nt - N_refine ** d : num_points_new_nt]  = h_child
        h_new[num_points_new_nt:] = h_pt
        set_dict['h'] = h_new

        nt_new = np.empty([num_points_new_nt , d])
        nt_new[:num_points_new_nt - N_refine ** d] = np.delete(nt, refined_point_index, axis= 0)
        nt_new[num_points_new_nt - N_refine ** d:num_points_new_nt] = child_nodes
        set_dict['nt'] = nt_new

        design_cells_nt_new= np.delete(design_cells_nt, refined_point_index, axis= 0)
        design_cells_new[:num_points_new_nt - N_refine ** d] = design_cells_nt_new
        design_cells_new[num_points_new_nt - N_refine ** d : num_points_new_nt] = design_cells_child
        design_cells_new[num_points_new_nt : ] = design_cells_pt
        set_dict['design_cells'] = design_cells_new

        mean_nt_new= np.delete(mean_nt, refined_point_index, axis= 0)
        mean_new[: num_points_new_nt - N_refine ** d] = mean_nt_new
        mean_new[num_points_new_nt - N_refine **d : num_points_new_nt ] = mean_child
        mean_new[num_points_new_nt: ] = mean_pt
        set_dict['mean'] = mean_new

        std_nt_new= np.delete(std_nt, refined_point_index, axis= 0)
        std_new[: num_points_new_nt - N_refine ** d] = std_nt_new
        std_new[num_points_new_nt - N_refine **d : num_points_new_nt ] = std_child
        std_new[num_points_new_nt :] = std_pt
        set_dict['std'] = std_new

        Vh_nt_new= np.delete(Vh_nt, refined_point_index, axis= 0)
        Vh_new[: num_points_new_nt - N_refine ** d] = Vh_nt_new
        Vh_new [num_points_new_nt - N_refine **d : num_points_new_nt ] =Vh_child
        Vh_new[num_points_new_nt :] = Vh_pt
        set_dict['Vh'] = Vh_new

        objective_cells_nt_new= np.delete(objective_cells_nt, refined_point_index, axis= 0)
        objective_cells_new[: num_points_new_nt - N_refine ** d] = objective_cells_nt_new
        objective_cells_new [num_points_new_nt - N_refine **d : num_points_new_nt ] =objective_cells_child
        objective_cells_new[num_points_new_nt: ] = objective_cells_pt
        set_dict['objective_cells'] = objective_cells_new

        points_new[:num_points_new_nt] = nt_new
        points_new[num_points_new_nt:] = pt
        set_dict['points'] = points_new

    else:
        num_points_new_pt = pt_length - 1 + N_refine ** d

        h_pt_new= np.delete(h_pt, p_index, axis= 0)
        h_new[:nt_length] = h_nt
        h_new[nt_length: -N_refine ** d] = h_pt_new
        h_new[-N_refine ** d:] = h_child
        set_dict['h'] = h_new

        pt_new = np.empty([num_points_new_pt , d])
        pt_new[: num_points_new_pt - N_refine ** d] = np.delete(pt, p_index, axis= 0)
        pt_new[num_points_new_pt - N_refine ** d:] = child_nodes

        design_cells_pt_new= np.delete(design_cells_pt, p_index, axis= 0)
        design_cells_new[: nt_length] = design_cells_nt
        design_cells_new[nt_length: - N_refine ** d] = design_cells_pt_new
        design_cells_new[- N_refine ** d:] = design_cells_child
        set_dict['design_cells'] = design_cells_new

        mean_pt_new= np.delete(mean_pt, p_index, axis= 0)
        mean_new[: nt_length] = mean_nt
        mean_new[nt_length: -N_refine ** d] = mean_pt_new
        mean_new[-N_refine ** d :] = mean_child
        set_dict['mean'] = mean_new

        std_pt_new= np.delete(std_pt, p_index, axis= 0)
        std_new[: nt_length] = std_nt
        std_new[nt_length: -N_refine ** d] = std_pt_new
        std_new[-N_refine ** d :] = std_child
        set_dict['std'] = std_new

        Vh_pt_new= np.delete(Vh_pt, p_index, axis= 0)
        Vh_new[: nt_length] = Vh_nt
        Vh_new[nt_length: -N_refine ** d] = Vh_pt_new
        Vh_new[-N_refine ** d :] = Vh_child
        set_dict['Vh'] = Vh_new

        objective_cells_pt_new= np.delete(objective_cells_pt, p_index, axis= 0)
        objective_cells_new[: nt_length] = objective_cells_nt
        objective_cells_new[nt_length: -N_refine ** d] = objective_cells_pt_new
        objective_cells_new[-N_refine ** d :] = objective_cells_child
        set_dict['objective_cells'] = objective_cells_new

        points_new[: nt_length] =  nt
        points_new[nt_length:] = pt_new
        set_dict['points'] = points_new

    return set_dict


def test_refine():

    import GPy
    set_dict= {}
    d= 2
    N_refine= 2
    nt_length = 10
    pt_length = 5
    obj_dim = 2

    set_dict['design_cells_nt'] = np.ones([nt_length , 2*d])
    for i in range(d):
        set_dict['design_cells_nt'][:, 2*i+1]= np.zeros([nt_length , ])

    set_dict['design_cells_pt'] = np.ones([pt_length, 2*d])
    for i in range(d):
        set_dict['design_cells_pt'][:, 2*i+1]= np.zeros([pt_length,])

    set_dict['design_cells']= np.append(set_dict['design_cells_nt'], set_dict['design_cells_pt'], axis=0)

    kernels= []
    noises = []
    for i in range(obj_dim):
        kernel = GPy.kern.RBF(1)
        kernels.append(kernel)
        noises.append(1e-10)

    set_dict['h_nt']= np.arange(nt_length ).reshape(-1, 1)
    set_dict['h_pt']= np.arange(pt_length).reshape(-1, 1)
    set_dict['h']= np.append(set_dict['h_nt'], set_dict['h_pt'], axis=0)

    set_dict['nt']= np.ones([nt_length ,d])*0.5
    set_dict['pt']= np.ones([pt_length,d])*0.5
    set_dict['points']= np.append(set_dict['nt'], set_dict['pt'], axis=0)

    set_dict['mean'] = np.random.rand(nt_length + pt_length, obj_dim)
    set_dict['std'] = np.random.rand(nt_length + pt_length, obj_dim)
    set_dict['XT'] = np.random.rand(nt_length + pt_length, obj_dim)
    set_dict['YT'] = np.random.rand(nt_length + pt_length, d)
    set_dict['Vh'] = np.random.rand(nt_length + pt_length, obj_dim)

    objective_cells_l = np.random.rand(nt_length + pt_length, obj_dim)
    objective_cells_u = objective_cells_l + np.abs(np.random.rand(nt_length + pt_length , obj_dim))
    set_dict['objective_cells'] = np.empty([nt_length  + pt_length, 0])
    for i in range(obj_dim):
        set_dict['objective_cells'] = np.append(set_dict['objective_cells'] , objective_cells_u[: , i].reshape(-1 , 1) , axis = 1)
        set_dict['objective_cells'] = np.append(set_dict['objective_cells'] , objective_cells_l[: , i].reshape(-1 , 1) , axis = 1)

    refined_index= 12

    number_of_evaluations = 1
    import experiment_setup
    rho = experiment_setup.rho
    alpha = experiment_setup.alpha
    delta = 1 - experiment_setup.confidence_ratio

    beta = np.zeros([obj_dim, 1])
    for i in range(obj_dim):
        beta[i, 0] = 3

    set_dict_copy = set_dict.copy()
    set_dict_new= refine_nodes(set_dict, refined_index , nt_length, d, N_refine, kernels, noises, number_of_evaluations,
                 rho, alpha, delta, beta)
