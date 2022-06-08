# Pareto Active Learning with Gaussian Processes and Adaptive Discretization
This repository contains the implementation of Adaptive ePAL algorithm from the paper ["Pareto Active Learning with Gaussian Processes and Adaptive Discretization"](https://arxiv.org/abs/2006.14061). 
Users can modify the "func_list" and "kernel_list" parameters in "main.py" file and create a setup file inside the "example_setups" directory similar to already existing files therein. Function names should match the file names in the "example_setups" directory. The results of the experiments can be found in the "Results" directory. 

## Required Packages
- GPy
- h5py

## Citation
If you use this library in an academic work, please cite our work "Pareto Active Learning with Gaussian Processes and Adaptive Discretization", Andi Nika, Kerem Bozgan, Sepehr Elahi, Çağın Ararat, Cem Tekin.
