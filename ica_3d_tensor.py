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

import pickle

count_dataset_path = ["/home/yulia/mnt/predict_mutations/tumour_mutation_counts_dataset.part1.pickle",
                        "/home/yulia/mnt/predict_mutations/tumour_mutation_counts_dataset.part2.pickle",
                        "/home/yulia/mnt/predict_mutations/tumour_mutation_counts_dataset.part3.pickle",
                         "/home/yulia/mnt/predict_mutations/tumour_mutation_counts_dataset.part4.pickle",
                          "/home/yulia/mnt/predict_mutations/tumour_mutation_counts_dataset.part5.pickle",
                           "/home/yulia/mnt/predict_mutations/tumour_mutation_counts_dataset.part6.pickle"]

model_params = "model.mut_counts.pickle"

def load_pickle(filename):
    with open(filename, 'rb') as pkl_file:
        filecontent = pickle.load(pkl_file)
    return filecontent

def dump_pickle(data, filename):
    with open(filename, 'wb') as pkl_file:
        pickle.dump(data, pkl_file)

def load_dataset(dataset_path):
    dataset, tumour_names = load_pickle(dataset_path[0])
    for path in dataset_path[1:]:
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
        print("Log likelihood {0:.3f}".format(- self.objective(params)))

    def objective(self,params):
        return - self.logprob(params)


if __name__ == "__main__":
    rs = npr.RandomState(0)

    print("Loading dataset...")
    data, tumour_names = load_dataset(count_dataset_path)

    data = data[:100,:,:]
    #data = np.random.poisson(lam=10,size = (dimensions['sample'], dimensions['region'], dimensions['type']))

    print("Optimizing...")

    dimensions = {'sample': data.shape[0], 'region': data.shape[1], 'type': data.shape[2]}
    latent_dimension = 10

    td = tensorDecomposition(data, dimensions, latent_dimension)

    num_latent_params = sum([d * latent_dimension for d in dimensions.values()])
    init_params = rs.randn(num_latent_params)

    minimize(value_and_grad(td.objective), td.params, method='Newton-CG', 
        jac=True, callback= td.callback, options={'maxiter':25})

    dump_pickle(td.params, "model_params")

    # compute average error
    # does lambda differ across tumours, types, etc.?
    # make training and test sets

