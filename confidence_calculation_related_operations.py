import numpy as np
import GPy
import experiment_setup
from numpy import sqrt , log , power , pi , maximum
import sys
import os
from util import pickle_read


class gp_model:
    def __init__(self, input_dim, obj_dim, kernel_type, noisy_observations, gp_sample, noise_mod_std):
        self.kernel_list= []
        self.noise_var_list= []
        self.noisy_observations= noisy_observations
        self.noise_std= noise_mod_std

        if gp_sample:
            for i in range(obj_dim):

                if kernel_type== 'rbf':
                    kernel= GPy.kern.RBF(input_dim, lengthscale=experiment_setup.lengthscale_list[i],
                                              variance= experiment_setup.variance_list[i])
                elif kernel_type == 'matern52':
                    kernel= GPy.kern.Matern52(input_dim, lengthscale=experiment_setup.lengthscale_list[i],
                                              variance= experiment_setup.variance_list[i])
                elif kernel_type == 'matern32':
                    kernel= GPy.kern.Matern32(input_dim, lengthscale=experiment_setup.lengthscale_list[i],
                                              variance= experiment_setup.variance_list[i])
                elif kernel_type == 'matern12':
                    kernel= GPy.kern.Matern12(input_dim, lengthscale=experiment_setup.lengthscale_list[i],
                                              variance= experiment_setup.variance_list[i])

                self.kernel_list.append(kernel)
                if noise_mod_std[0, i] < np.sqrt(0.2 * 1e-5):
                    self.noise_var_list.append(0.2 * 1e-5)
                else:
                    self.noise_var_list.append(np.power(noise_mod_std[0, i], 2))

        elif experiment_setup.use_saved_hypers:
            self.kernel_list = []
            self.noise_var_list = []
            self.noisy_observations = noisy_observations

            sys.path.append('pretrained_hyperparameters')
            hyper_path= os.path.join('pretrained_hyperparameters', 'hyperparameters')
            hyper_dict= pickle_read(hyper_path)
            for i, funct in enumerate(experiment_setup.function_list):
                ls= hyper_dict[funct]['setup1'][str(experiment_setup.discretization_hyper)]['ls']
                var= hyper_dict[funct]['setup1'][str(experiment_setup.discretization_hyper)]['var']
                noise= hyper_dict[funct]['setup1'][str(experiment_setup.discretization_hyper)]['noise']
                kernel_type= hyper_dict[funct]['setup1'][str(experiment_setup.discretization_hyper)]['kernel_type']
                if kernel_type== 'rbf':
                    self.kernel_list.append(GPy.kern.RBF(input_dim, lengthscale=ls,
                                          variance=var))
                elif kernel_type== 'matern52':
                    self.kernel_list.append(GPy.kern.Matern52(input_dim, lengthscale=ls,
                                          variance=var))
                if kernel_type== 'matern32':
                    self.kernel_list.append(GPy.kern.Matern32(input_dim, lengthscale=ls,
                                          variance=var))
                if kernel_type== 'matern12':
                    self.kernel_list.append(GPy.kern.Matern12(input_dim, lengthscale=ls,
                                          variance=var))
                self.noise_var_list.append(noise)

        self.XT= np.zeros([0, input_dim])
        self.YT= np.zeros([0, obj_dim])
        self.input_dim= input_dim
        self.obj_dim= obj_dim

    def add_sample(self, xt, yt):
        self.XT= np.append(self.XT, xt, axis=0)
        self.YT= np.append(self.YT, yt, axis=0)

    def update(self):
        self.model_list= []
        for i in range(self.obj_dim):
            model= GPy.models.GPRegression(self.XT, self.YT[:, i].reshape(-1,1), kernel= self.kernel_list[i])
            model.Gaussian_noise = self.noise_var_list[i]
            self.model_list.append(model)

    def predict(self, X):
        mean= np.zeros([X.shape[0], self.obj_dim])
        std= np.zeros([X.shape[0], self.obj_dim])

        for i in range(self.obj_dim):
            mean_, var_ = self.model_list[i].predict_noiseless(X)
            mean[:,i]= mean_[:,0]
            std[:, i]= np.sqrt(var_[:, 0])

        return mean, std

def gp_prediction(kernels, points, XT, YT, noises):

    Mean= np.zeros([points.shape[0], experiment_setup.objective_dim])
    Std= np.zeros([points.shape[0], experiment_setup.objective_dim])

    for i in range(len(kernels)):
        kernel= kernels[i]
        model = GPy.models.GPRegression(XT, YT[:, i].reshape(-1, 1), kernel=kernel)
        if not experiment_setup.noisy_observations:
            model.Gaussian_noise = (10 ** (-10))
        else:
            model.Gaussian_noise= noises[i]
        mean, var = model.predict_noiseless(points)
        std = np.sqrt(var)
        Mean[:, i] = mean[:, 0]
        Std[:, i] = std[:, 0]
    return Mean, Std

def test_gp_prediction():
    import time
    kernels = []
    noises = []
    for i in range(2):
        kernels.append(GPy.kern.RBF(input_dim = 2))
        noises.append(1e-10)

    points = np.random.rand(10000 , 2)
    XT = np.random.rand(50 , 2)
    YT = np.random.rand(50 , 2)

    start1 = time.time()
    Mean, Std = gp_prediction(kernels, points, XT, YT, noises)
    stop1 = time.time()

    print(stop1 - start1)

def gp_prediction_gpflow(kernels, points, XT, YT, noises):

    Mean= np.zeros([points.shape[0], experiment_setup.objective_dim])
    Std= np.zeros([points.shape[0], experiment_setup.objective_dim])

    for i in range(len(kernels)):
        kernel= kernels[i]
        model = gpflow.models.GPR((XT,  YT[:, i].reshape(-1, 1)), kernel= kernel)

        if not experiment_setup.noisy_observations:
            model.likelihood.variance.assign(10 ** (-6))
        else:
            model.Gaussian_noise= noises[i]
            model.likelihood.variance.assign(noises[i])

        mean, var = model.predict_f(points)
        std = np.sqrt(var)
        Mean[:, i] = mean[:, 0]
        Std[:, i] = std[:, 0]
    return Mean, Std

def test_gp_prediction_gpflow():
    import time
    kernels = []
    noises = []
    for i in range(2):
        kernels.append(gpflow.kernels.RBF())
        noises.append(1e-5)

    points = np.random.rand(10000 , 2)
    XT = np.random.rand(50 , 2)
    YT = np.random.rand(50 , 2)

    start1 = time.time()
    Mean, Std = gp_prediction_gpflow(kernels, points, XT, YT, noises)
    stop1 = time.time()

    print(stop1 - start1)


def calculate_vh(kernels, rho, alpha, delta, N_nodes, m, d, h, N_refine , v1, kernel_type):

    lengthscales= []
    variances= []
    for kernel in kernels:
        lengthscales.append(kernel.lengthscale.values[0])
        variances.append(kernel.variance.values[0])

    D1= d
    N= N_refine
    Vh= np.zeros([N_nodes, m])
    diamX = sqrt(D1)
    depth= h.copy()

    for i in range(m):
        if kernel_type== 'rbf':
            Cki = sqrt(variances[i]) / lengthscales[i]
        elif kernel_type=='matern52':
            Cki= sqrt(10) * sqrt(variances[i]) / lengthscales[i]
        elif kernel_type=='matern32':
            Cki= sqrt(6) * sqrt(variances[i]) / lengthscales[i]
        elif kernel_type=='matern12':
            Cki= sqrt(2) * sqrt(variances[i]) / sqrt(lengthscales[i])

        C1 = power(sqrt(D1)+ 1 , D1) * power(Cki, 1/alpha) * power(diamX, D1) * power(1 /2, D1)
        C2 = 2 * log(2 * power(C1 , 2) * power(pi , 2) / 6 )
        C3 = 1. + 2.7 * sqrt(2 * D1 * alpha * log(2))

        Vh[:, i] = 4 * Cki * power((v1 * power(rho , depth[:, 0])) , alpha) *\
                   (sqrt(C2 + 2 * log(2 * power(depth[:, 0] + 1, 2) * power(pi , 2) * m / (6 * delta)) + \
                     depth[:, 0] * log(N) +\
                         maximum(0, -4 * D1/alpha * log(Cki * power(v1 * power(rho , depth[: , 0]) , alpha)))) +C3)


    return Vh


def update_objective_cells(mean, std, Vh, beta, flag_hmax , refined = False):

    if flag_hmax:

        objective_cells= np.zeros([mean.shape[0], 2*experiment_setup.objective_dim])
        for i in range(experiment_setup.objective_dim):
            objective_cells_u=  mean[:,i] + beta[i,0] ** (1 / 2) * std[:,i]
            objective_cells_l=  mean[:,i] -beta[i,0] ** (1 / 2) * std[:,i]
            objective_cells[:,2*i]= objective_cells_u
            objective_cells[:, 2*i+1]= objective_cells_l
    else:
        objective_cells= np.zeros([mean.shape[0], 2*experiment_setup.objective_dim])
        for i in range(experiment_setup.objective_dim):
            objective_cells_u=  mean[:,i] + beta[i,0] ** (1 / 2) * std[:,i] + Vh[:,i]
            objective_cells_l=  mean[:,i] -beta[i,0] ** (1 / 2) * std[:,i] - Vh[:,i]
            objective_cells[:,2*i]= objective_cells_u
            objective_cells[:, 2*i+1]= objective_cells_l
    return objective_cells

def test_update_objective_cells():
    mean= np.random.normal(0.0, 1.0, (10,2))
    std = np.random.normal(0.0, 1.0, (10, 2))
    Vh = np.random.normal(0.0, 1.0, (10, 2))
    beta= 3.0
    objective_cells= update_objective_cells(mean, std, Vh, beta)
    print(objective_cells.shape)