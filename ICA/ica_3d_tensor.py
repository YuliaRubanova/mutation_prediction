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

#count_dataset_path = ["/Users/yulia/Documents/mutational_signatures/mutation_prediction_data/tumour_mutation_counts_dataset.small.pickle"]
model_params_file = "trained_models/model.mut_counts.lat{}.loss_{}.pickle"

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

# def bernoulli_pmf(k, log_prob):
#     return np.multiply(log_prob, k) + np.multiply(np.log(1 - np.exp(log_prob)), (1-k))

class tensorDecomposition():
    def __init__(self, data, data_dimensions, latent_dimension, loss_type):
        self.dimensions = data_dimensions
        self.latent_dimension = latent_dimension
        self.data = data

        num_latent_params = sum([d * latent_dimension for d in dimensions.values()])
        self.params = rs.randn(num_latent_params) / 2
        self.iter = 0
        self.loss_type = loss_type

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
        if loss_type == "poisson":
            prob_mass = poisson_pmf(self.data, self.get_lambda(params)) /  self.dimensions['sample'] / self.dimensions['region'] / self.dimensions['type']
        else:
            prob_mass = -np.sqrt(np.sum((self.data - np.exp(self.get_lambda(params)))**2))
        return np.sum(prob_mass) 

    def callback(self, params):
        self.iter += 1
        print("{}: Log likelihood {:.3f}".format(self.iter, - self.objective(params)))
        
        model_params_file_tmp = "trained_models/model.mut_counts.lat{}.small.tmp.pickle".format(self.latent_dimension, self.loss_type)
        dump_pickle(self.params, model_params_file_tmp)

    def objective(self,params):
        self.params = params
        return - self.logprob(params)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train model to predict mutation rates per region')
    parser.add_argument('--test', help='test mode: only read trained params and analyze', action="store_true")
    parser.add_argument('--latents', help='number of latent dimensions', default=100, type=int)
    parser.add_argument('-n', '--tumours', help='number of tumours to include in the set', default=None)
    parser.add_argument('--loss', help = "loss type: poisson or mean_squared", default="poisson")

    args = parser.parse_args()
    test_mode = args.test
    n_tumours = args.tumours
    latent_dimension = args.latents
    loss_type = args.loss

    rs = npr.RandomState(0)

    print("Loading dataset...")
    data, tumour_names = load_dataset(count_dataset_path)

    if n_tumours is not None:
        data = data[:int(n_tumours),:,:]
    else:
        n_tumours = data.shape[0]

    print("Processing {} tumours with {} latent dimensions...".format(data.shape[0], latent_dimension))

    dimensions = {'sample': data.shape[0], 'region': data.shape[1], 'type': data.shape[2]}
    
    model_params_file = model_params_file.format(latent_dimension, loss_type)

    td = tensorDecomposition(data, dimensions, latent_dimension, loss_type)

    num_latent_params = sum([d * latent_dimension for d in dimensions.values()])
    
    if not test_mode:
        print("Optimizing...")
        init_params = rs.randn(num_latent_params)

        minimize(value_and_grad(td.objective), td.params, method='BFGS', 
            jac=True, callback= td.callback, options={'maxiter':100})

        dump_pickle(np.array(td.params), model_params_file)

        print("Likelihood")
        print(td.logprob(td.params) / dimensions['sample'] / dimensions['region'] / dimensions['type'])
    else:
        if not os.path.isfile(model_params_file):
            print("Model parameters not found: " + model_params_file)
            exit()

        trained_params = load_pickle(model_params_file)
        print(td.params[:10])
        print(trained_params.shape)

        log_lam = td.get_lambda(trained_params)
        print(log_lam.shape)
        print(np.exp(log_lam)[0,:,0])
        print(np.exp(log_lam)[:,0,0])
        print(np.exp(log_lam)[0,0,])
        print("Mean squared error and mean difference:")
        print(np.sqrt(np.mean((data - np.exp(log_lam))**2)))
        print("Where true mutation occurred:" + str(np.mean((data - np.exp(log_lam))[data == 1])))
        print("Where no mutation occurred:" + str(np.mean((data - np.exp(log_lam))[data == 0])))
        print("Mean lambda")
        print("Where true mutation occurred:" + str(np.mean((np.exp(log_lam))[data == 1])))
        print("Where no mutation occurred:" + str(np.mean((np.exp(log_lam))[data == 0])))


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
        exit()

        sample_latents, region_latents, type_latents = td.unpack_params(trained_params)
        print(sample_latents)
        print(type_latents)

        #compare sample and region latents
        #can it overfit?
        # make training and test sets
        # predict lambda for new tumour, region


