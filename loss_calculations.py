import numpy as np
from util import denormalizeX
from util import normalize_minmax_minus_one_one
from pygmo import hypervolume

def get_approximate_pareto_set_and_front(X, Y):

    discard = np.zeros([X.shape[0], 1])

    for i in range(X.shape[0]):
        for j in range(i+1, X.shape[0]):

            vi = Y[i, :].reshape(-1, 1)
            vj = Y[j, :].reshape(-1, 1)

            if np.all(vi > vj):
                discard[j, 0] = 1

            if np.all(vj > vi):
                discard[i, 0] = 1
                break

    pareto_set = X[np.where(discard == 0)[0], :]
    pareto_front = Y[np.where(discard == 0)[0], :]

    return pareto_set, pareto_front

# TODO rewrite the code for finding pareto front and set
def get_pareto_front_set(x, y, get_indeces= False):

    indeces = np.arange(y.shape[0])
    next_point_index = 0  # Next index in the is_efficient array to search for

    while next_point_index<len(y):
        nondominated = np.any(y>y[next_point_index], axis=1)
        nondominated[next_point_index] = True
        indeces = indeces[nondominated]  # Remove dominated points
        y =y[nondominated]
        next_point_index = np.sum(nondominated[:next_point_index])+1
    x= x[indeces]

    if get_indeces:
        return x, y, indeces
    else:
        return x, y

def get_pareto_front_set_special_m2(x , y , get_indeces = False):
    n_for = y.shape[0]
    kept_indeces_true_false = np.ones(n_for , dtype = bool)

    index_map_sort = np.argsort(y[:, 0])[::-1]
    obj2 = y[: , 1][index_map_sort]

    accumulated_max = np.zeros([n_for])
    accumulated_max[0] = -np.inf
    accumulated_max[1:] = (np.maximum.accumulate(obj2))[:-1]
    kept_indeces_sorted_true_false = obj2 > accumulated_max
    kept_indeces_true_false[index_map_sort] = kept_indeces_sorted_true_false
    y_opt = y[kept_indeces_true_false]
    x_opt = x[kept_indeces_true_false]

    if get_indeces:
        indeces = np.arange(n_for)[kept_indeces_true_false]
        return x_opt , y_opt , indeces
    else:
        return x_opt , y_opt


def get_pareto_front_set_special_m2_2(x , y , get_indeces = False):

    index_map_sort = np.argsort(-y[:, 0]) #map sorted array elements to their original order in the source array

    y_sorted= y[index_map_sort]

    n_for = y.shape[0]

    kept_indeces_sorted_true_false = np.ones([n_for], dtype = bool)
    kept_indeces_true_false = np.ones([n_for], dtype = bool)

    current_max = -np.inf

    for i in range(n_for):
        yi_obj2 = y_sorted[i, :][1]
        if yi_obj2 <= current_max:
            kept_indeces_sorted_true_false[i] = False

        else:
            current_max = yi_obj2

    kept_indeces_true_false[index_map_sort[np.arange(n_for)]] = kept_indeces_sorted_true_false
    y_opt = y[kept_indeces_true_false]
    x_opt = x[kept_indeces_true_false]

    if get_indeces:
        indeces = np.arange(n_for)[kept_indeces_true_false]
        return x_opt , y_opt , indeces

    else:
        return x_opt , y_opt

def test_get_pareto_front_set_special():

    count1 = 0
    count_valid = 0
    count3 = 0
    for i in range(100):
        x = np.random.rand(1048576 , 2)
        y = np.random.rand(1048576, 2)
        import time

        start1 = time.time()
        x1 , y1 , indeces1  = get_pareto_front_set_special_m2(y , x , get_indeces = True)
        stop1 = time.time()

        start2 = time.time()
        x2 , y2 , indeces2  = get_pareto_front_set_special_m2_2(y , x , get_indeces = True)
        stop2 = time.time()

        start3 = time.time()
        x3 , y3 , indeces3 = get_pareto_front_set(y , x , get_indeces = True)
        stop3 = time.time()
        print()
        print(f'special algorithm run time: {stop1 - start1}')
        # print(f'special algorithm2 run time: {stop2 - start2}')
        print(f'unoptimized algorithm run time: {stop3 - start3}')

        print(f'validate indeces 1 equal indeces 2 : {np.all(indeces1 == indeces2) and np.all(indeces2 == indeces3)}')
        count_valid += np.all(indeces1 == indeces2) and np.all(indeces2 == indeces3)
        if stop1 - start1 <= stop3 - start3:
            count1 += 1
        else:
            count3 += 1
    print()
    print()
    print(f'number of wins1: {count1}, wins2: {count3}')
    print(f'number of valid results = {count_valid}')


def test_get_pareto_front_set():
    import matplotlib.pyplot as plt
    objective_array= np.random.normal(0.0, 1.0, (100,2))
    design_array= np.random.normal(0.0, 1.0, (100,2))
    upper_pareto_set, upper_pareto_front, indeces= get_pareto_front_set(design_array, objective_array, get_indeces= True)

    plt.figure()
    plt.scatter(objective_array[:,0], objective_array[:, 1], color= 'black', alpha= 0.5)

    plt.scatter(upper_pareto_front[:,0], upper_pareto_front[:,1], color= 'red', alpha=0.5)
    plt.show()

# TODO rewrite the code for finding pareto front
def get_pareto_front(y, get_indeces=False):
    indeces = np.arange(y.shape[0])
    next_point_index = 0  # Next index in the is_efficient array to search for
    while next_point_index<len(y):
        nondominated = np.any(y>y[next_point_index], axis=1)
        nondominated[next_point_index] = True
        indeces = indeces[nondominated]  # Remove dominated points
        y =y[nondominated]
        next_point_index = np.sum(nondominated[:next_point_index])+1

    if get_indeces:
        return y, indeces
    else:
        return y

def test_get_pareto_front():
    import matplotlib.pyplot as plt
    objective_array= np.random.normal(0.0, 1.0, (100,2))
    plt.figure()
    plt.scatter(objective_array[:,0], objective_array[:, 1], color= 'black', alpha= 0.5)
    lower_pareto_front= get_pareto_front(-objective_array)
    plt.scatter(-lower_pareto_front[:,0], -lower_pareto_front[:,1], color= 'red', alpha=0.5)
    plt.show()

def get_lower_pareto_front(design_points_normalized, miny, maxy, minx, scalex, objective_function):
    design_points = denormalizeX(design_points_normalized, scalex, minx)
    objective_points= objective_function(design_points)
    objective_points_normalized=  normalize_minmax_minus_one_one(objective_points, miny, maxy)
    objective_points_normalized_minus= -objective_points_normalized.copy()
    lower_pareto_front_minus= get_pareto_front(objective_points_normalized_minus)
    lower_pareto_front_normalized= -lower_pareto_front_minus
    return lower_pareto_front_normalized

def get_upper_pareto_front(design_points_normalized, miny, maxy, minx, scalex, objective_function):
    design_points = denormalizeX(design_points_normalized, scalex, minx)
    objective_points= objective_function(design_points)
    objective_points_normalized=  normalize_minmax_minus_one_one(objective_points, miny, maxy)
    upper_pareto_front_normalized= get_pareto_front(objective_points_normalized)
    return upper_pareto_front_normalized

def test_get_upper_pareto_front():
    design_points_normalized= np.random.uniform(0.0, 1.0, (100,2))
    design_points= design_points_normalized

    import matplotlib.pyplot as plt
    def objective_function(X):
        x1 = X[:, 0].reshape(-1, 1)
        x2 = X[:, 1].reshape(-1, 1)

        x1_branin = (x1 * 15) + np.array([[-5]])
        x2_branin = (x2 * 15) + np.array([[0]])
        x1_six = (x1 * 6) + np.array([[-3]])
        x2_six = (x2 * 4) + np.array([[-2]])

        y1 = -(np.square(x2_branin - (5.1 / (4 * np.square(np.pi))) * np.square(x1_branin) +
                         (5 / np.pi) * x1_branin - 6) + 10 * (1 - (1. / (8 * np.pi))) * np.cos(x1_branin) + 10)

        y2 = -((4 - 2.1 * x1_six ** 2 + x1_six ** 4 / 3) * x1_six ** 2 + x1_six * x2_six + (
                    -4 + 4 * x2_six ** 2) * x2_six ** 2)

        y = np.append(y1, y2, axis=1)

        return y
    maxy= np.array([ 0.45762169,  0.9216])
    miny= np.array([-308.12909601, -162.9])
    minx= np.array([0.0, 0.0])
    maxx= np.array([1.0, 1.0])
    scalex= maxx-minx

    objective_points= objective_function(design_points)
    upper_pareto_front= get_upper_pareto_front(design_points_normalized, miny, maxy, minx, scalex, objective_function)
    fig, ax= plt.subplots(1)
    ax.scatter(objective_points[:, 0], objective_points[:, 1], color='orange')
    ax.scatter(upper_pareto_front[:, 0], upper_pareto_front[:, 1], color='red')
    plt.show()

def get_lower_pareto_front_disc(objective_points_normalized):

    objective_points_normalized_minus= -objective_points_normalized.copy()
    lower_pareto_front_minus= get_pareto_front(objective_points_normalized_minus)
    lower_pareto_front_normalized= -lower_pareto_front_minus
    return lower_pareto_front_normalized

def test_get_lower_pareto_front():

    def sample_func(X):
        y1 = np.abs(10 * X[:, 0].reshape(-1, 1))
        y2 = np.abs(10 * X[:, 1].reshape(-1, 1))
        return np.append(y1, y2, axis=1)


    design_points_normalized= np.random.uniform(0.0, 1.0, (10, 2))
    # true_pareto_front_normalized = np.random.uniform(-1.0, 1.0, (10, 2))
    minx = 0.0
    maxx = 1.0
    scalex = 1.0
    miny = -1.0
    maxy = 1.0
    scaly = 2.0
    design_points= denormalizeX(design_points_normalized, scalex, minx)
    objective_points= sample_func(design_points)
    print('objective points')
    print(objective_points)
    objective_points_normalized= normalize_minmax_minus_one_one(objective_points, miny, maxy)
    print('objective_points normalized')
    print(objective_points_normalized)
    lower_pareto_front_normalized = get_lower_pareto_front(design_points_normalized, miny, maxy, minx, scalex, sample_func)
    print('lower_pareto_fornt_normalized')
    print(lower_pareto_front_normalized)

def get_loss(predicted_pareto_set_normalized, true_pareto_front_normalized, miny, maxy, minx, scalex, objective_function):

    lower_pareto_front_normalized= get_lower_pareto_front(predicted_pareto_set_normalized, miny, maxy,  minx, scalex, objective_function)

    min_distances = np.zeros([true_pareto_front_normalized.shape[0], 1])

    for i in range(true_pareto_front_normalized.shape[0]):

        pi = true_pareto_front_normalized[i, :].reshape(-1, 1)
        min_dist_temp = np.inf

        for j in range(lower_pareto_front_normalized.shape[0]):

            plj= lower_pareto_front_normalized[j, :].reshape(-1,1)
            distij = np.linalg.norm(pi- plj)

            if distij < min_dist_temp:
                min_dist_temp = distij.copy()

        min_distances[i, 0]= min_dist_temp.copy()

    pred_error = np.mean(min_distances)

    return pred_error

def get_loss_disc(objective_points_normalized,true_pareto_front_normalized):

    lower_pareto_front_normalized= get_lower_pareto_front_disc(objective_points_normalized)

    min_distances = np.zeros([true_pareto_front_normalized.shape[0], 1])

    for i in range(true_pareto_front_normalized.shape[0]):

        pi = true_pareto_front_normalized[i, :].reshape(-1, 1)
        min_dist_temp = np.inf

        for j in range(lower_pareto_front_normalized.shape[0]):

            plj= lower_pareto_front_normalized[j, :].reshape(-1,1)
            distij = np.linalg.norm(pi- plj)

            if distij < min_dist_temp:
                min_dist_temp = distij.copy()

        min_distances[i, 0]= min_dist_temp.copy()

    pred_error = np.mean(min_distances)

    return pred_error

def test_get_loss():

    def sample_func(X):
        y1= np.abs(10*X[:, 0].reshape(-1,1))
        y2= np.abs(10*X[:, 1].reshape(-1,1))
        return np.append(y1, y2, axis=1)

    predicted_pareto_set_normalized= np.random.uniform(0.0, 1.0, (10,2))
    true_pareto_front_normalized= np.random.uniform(-1.0, 1.0, (10,2))
    minx= 0.0
    maxx= 1.0
    scalex= 1.0
    miny= -1.0
    maxy= 1.0
    scaly= 2.0
    pred_error = get_loss(predicted_pareto_set_normalized, true_pareto_front_normalized, miny, maxy, minx, scalex, sample_func)

    print('pred error', pred_error)



def get_hypervolume(predicted_pareto_set_n, objective_func, miny, maxy, minx, scalex, objective_dim, app_pareto_front_n, epsilon):

    lower_pareto_front_normalized= get_lower_pareto_front(predicted_pareto_set_n, miny, maxy, minx, scalex, objective_func)

    ref= np.ones([1, objective_dim])*(-1)-10e-4

    hv= hypervolume(-lower_pareto_front_normalized)
    hyv= hv.compute(-ref[0, :])
    print('hyper volume of predicted set:')
    print(hyv)
    return hyv

def get_epsilon_coverage_rate(predicted_pareto_set_n, approximated_pareto_front_n, objective_func, scalex, minx, miny,
                              maxy, epsilon, diabetes=False, predicted_obj=None):

    if diabetes==False:
        predicted_pareto_set = denormalizeX(predicted_pareto_set_n, scalex, minx)
        predicted_pareto_front = objective_func(predicted_pareto_set)
    else:
        predicted_pareto_front= predicted_obj.copy()

    predicted_pareto_front_n= normalize_minmax_minus_one_one(predicted_pareto_front, miny, maxy)
    covered = np.zeros([1,approximated_pareto_front_n.shape[0]])
    for i in range(approximated_pareto_front_n.shape[0]):
        pareto_front_i =approximated_pareto_front_n[i, :].reshape(-1, 1)

        for j in range(predicted_pareto_front_n.shape[0]):
            predicted_pareto_foront_n_j =predicted_pareto_front_n[j, :].reshape(-1, 1)

            if np.all(predicted_pareto_foront_n_j + epsilon >= pareto_front_i):
                covered[0, i] = 1
                break

    print('number of covered points:{}'.format(np.sum(covered, axis=1)))
    print('total number of pareto points:', approximated_pareto_front_n.shape[0])
    covering_ratio = np.sum(covered, axis=1) /covered.shape[1]
    print('covering ratio:', covering_ratio)

    return covering_ratio

def test_epsilon_cov():
    import sys
    sys.path.append('.\\example_setups')
    import experiment_setup_branin_sixhump
    from util import get_grid, normalize_minmax_zero_one, normalize_minmax_minus_one_one
    path = "Results/branin_sixhump_16_noiseless\\parego\\parego_38_seed1.txt"
    objective_dim = 2
    epsilon_percent = 30
    objective_func = experiment_setup_branin_sixhump.objective_function
    miny = np.array(experiment_setup_branin_sixhump.y_bound[1])
    maxy = np.array(experiment_setup_branin_sixhump.y_bound[0])
    minx = np.array(experiment_setup_branin_sixhump.x_bound[1])
    maxx = np.array(experiment_setup_branin_sixhump.x_bound[0])
    X_grid = get_grid(experiment_setup_branin_sixhump.x_bound,
                      experiment_setup_branin_sixhump.discretization_pareto, experiment_setup_branin_sixhump.design_dim)


    scalex = maxx - minx

    Y_grid = objective_func(X_grid)
    X_grid_n = normalize_minmax_zero_one(X_grid, minx, maxx)
    Y_grid_n = normalize_minmax_minus_one_one(Y_grid, miny, maxy)
    app_pareto_set_n, app_pareto_front_n = get_approximate_pareto_set_and_front(X_grid_n,
                                                                                Y_grid_n)

    predicted_pareto_front_n = np.array([[0.75, 0.75], [0.8, 0.75]])
    predicted_pareto_set_n = np.array([[0.5483871 , 0.16129032], [0.51612903, 0.19354839]])


    epsilon = 0.01
    get_epsilon_coverage_rate(predicted_pareto_set_n, app_pareto_set_n, objective_func, scalex, minx, miny,
                              maxy,epsilon)

def get_epsilon_accuracy_rate(predicted_pareto_set_n, approximated_pareto_front_n, objective_func, scalex, minx, miny,
                              maxy, epsilon, diabetes= False, predicted_obj= None):

    if diabetes==False:
        predicted_pareto_set = denormalizeX(predicted_pareto_set_n, scalex, minx)
        predicted_pareto_front = objective_func(predicted_pareto_set)
    else:
        predicted_pareto_front= predicted_obj.copy()

    predicted_pareto_front_n= normalize_minmax_minus_one_one(predicted_pareto_front, miny, maxy)

    accurate= np.ones([1, predicted_pareto_front_n.shape[0]])

    for i in range(predicted_pareto_front_n.shape[0]):
        obj_i= predicted_pareto_front_n[i, :].reshape(-1,1)

        for j in range(approximated_pareto_front_n.shape[0]):
            pareto_j= approximated_pareto_front_n[j,:].reshape(-1,1)

            if np.all(pareto_j-epsilon>=obj_i):
                accurate[0, i]= 0
                break



    print('num accurate:', np.sum(accurate, axis= 1))
    print('total number of predictedpoints:',  predicted_pareto_front_n.shape[0])
    accuracy_ratio= np.sum(accurate, axis= 1)/predicted_pareto_front_n.shape[0]
    print('epsilon accuraty ratio:', accuracy_ratio)

    return  accuracy_ratio
