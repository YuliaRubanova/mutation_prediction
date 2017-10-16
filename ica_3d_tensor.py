from __future__ import absolute_import
from __future__ import print_function
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.optimizers import adam
import autograd.scipy.stats.t as t
from autograd import value_and_grad, grad

from scipy.optimize import minimize
from autograd.scipy.special import gammaln

import argparse
import pickle
import os

count_dataset_path = ["/home/yulia/mnt/mutation_prediction_data/tumour_mutation_counts_dataset.part1.pickle",
                        "/home/yulia/mnt/mutation_prediction_data/tumour_mutation_counts_dataset.part2.pickle",
                        "/home/yulia/mnt/mutation_prediction_data/tumour_mutation_counts_dataset.part3.pickle",
                         "/home/yulia/mnt/mutation_prediction_data/tumour_mutation_counts_dataset.part4.pickle",
                          "/home/yulia/mnt/mutation_prediction_data/tumour_mutation_counts_dataset.part5.pickle",
                           "/home/yulia/mnt/mutation_prediction_data/tumour_mutation_counts_dataset.part6.pickle"]

model_params_file = "trained_models/model.mut_counts.lat{}.pickle"

def load_pickle(filename):
    with open(filename, 'rb') as pkl_file:
        filecontent = pickle.load(pkl_file)
    return filecontent

def dump_pickle(data, filename):
    with open(filename, 'wb') as pkl_file:
        pickle.dump(data, pkl_file)

def load_dataset(dataset_path, n_parts = 1000):
    dataset, tumour_names = load_pickle(dataset_path[0])
    for path in dataset_path[1:n_parts]:
        dat, names = load_pickle(path)
        dataset = np.concatenate((dataset, dat))
        tumour_names.append(names)
    return dataset, tumour_names

def poisson_pmf(k, log_mu):
    return k * log_mu - gammaln(k + 1) - np.exp(log_mu)

class tensorDecomposition():
    def __init__(self, data, data_dimensions, latent_dimension):
        self.dimensions = data_dimensions
        self.latent_dimension = latent_dimension
        self.data = data

        num_latent_params = sum([d * latent_dimension for d in dimensions.values()])
        self.params = rs.randn(num_latent_params) / 2
        self.iter = 0

    def unpack_params(self, params):
        sample_latents_dim = self.dimensions['sample'] * self.latent_dimension
        region_latents_dim = self.dimensions['region'] * self.latent_dimension
        type_latents_dim = self.dimensions['type'] * self.latent_dimension

        sample_latents = np.reshape(params[:sample_latents_dim], (self.dimensions['sample'],latent_dimension))
        region_latents = np.reshape(params[sample_latents_dim:(sample_latents_dim+region_latents_dim)], (dimensions['region'],latent_dimension))
        type_latents = np.reshape(params[(sample_latents_dim+region_latents_dim):], (self.dimensions['type'],latent_dimension))

        return sample_latents, region_latents, type_latents


    def get_lambda(self,params):
        sample_latents, region_latents, type_latents = self.unpack_params(params)
        
        log_lam = np.dot((region_latents * sample_latents[:, np.newaxis]), type_latents.T)
        return log_lam

    def logprob(self, params):
        return np.sum(poisson_pmf(self.data, self.get_lambda(params))) / self.dimensions['sample'] / self.dimensions['region'] / self.dimensions['type']

    def callback(self, params):
        self.iter += 1
        print("{}: Log likelihood {0:.3f}".format(self.iter, - self.objective(params)))
        
        model_params_file_tmp = "trained_models/model.mut_counts.lat{}.tmp.pickle".format(self.latent_dimension)
        dump_pickle(self.params, model_params_file_tmp)

    def objective(self,params):
        return - self.logprob(params)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train model to predict mutation rates per region')
    parser.add_argument('--test', help='test mode: only read trained params and analyze', action="store_true")

    args = parser.parse_args()
    test_mode = args.test

    rs = npr.RandomState(0)

    print("Loading dataset...")
    data, tumour_names = load_dataset(count_dataset_path, 1) #!!!!!!

    data = data[:100,:,:] # !!!!!!
    #data = np.random.poisson(lam=10,size = (dimensions['sample'], dimensions['region'], dimensions['type']))

    dimensions = {'sample': data.shape[0], 'region': data.shape[1], 'type': data.shape[2]}
    latent_dimension = 100
    
    model_params_file = model_params_file.format(latent_dimension)

    td = tensorDecomposition(data, dimensions, latent_dimension)

    num_latent_params = sum([d * latent_dimension for d in dimensions.values()])
    
    if not test_mode:
        print("Optimizing...")
        init_params = rs.randn(num_latent_params)

        minimize(value_and_grad(td.objective), td.params, method='Newton-CG', 
            jac=True, callback= td.callback, options={'maxiter':100})

        dump_pickle(td.params, model_params_file)
    else:
        if not os.path.isfile(model_params_file):
            print("Model parameters not found: " + model_params_file)
            exit()

        trained_params = load_pickle(model_params_file)
        log_lam = td.get_lambda(trained_params)
        print(log_lam.shape)
        print(np.exp(log_lam)[0,:,0])
        print(np.exp(log_lam)[:,0,0])
        print(np.exp(log_lam)[0,0,])
        print("Mean and std of error:")
        print(np.sqrt(np.mean((data - np.exp(log_lam))**2)))
        print(np.std(data - np.exp(log_lam)))

        print("Mean lambda for tumour, region, type")
        print(np.mean(np.mean(np.exp(log_lam),axis=0)))
        print(np.mean(np.mean(np.exp(log_lam),axis=1)))
        print(np.mean(np.mean(np.exp(log_lam),axis=2)))

        print("Std of lambda for tumour, region, type")
        print(np.mean(np.std(np.exp(log_lam),axis=0)))
        print(np.mean(np.std(np.exp(log_lam),axis=1)))
        print(np.mean(np.std(np.exp(log_lam),axis=2)))

        print("Mean lambda with 1 type averaged across regions")
        print(np.mean(np.exp(log_lam)[:,:,0], axis=1))
        print("Mean lambda with 1 type averaged across tumours")
        print(np.mean(np.exp(log_lam)[:,:,0], axis=0))
        print("Likelihood")
        print(td.logprob(trained_params))

        sample_latents, region_latents, type_latents = td.unpack_params(trained_params)
        print(sample_latents)
        print(type_latents)

        #compare sample and region latents
        #experiments with diff latent dimentions
        #can it overfit?

    # compute average error
    # does lambda differ across tumours, types, etc.?
    # make training and test sets
    # predict lambda for new tumour, region


