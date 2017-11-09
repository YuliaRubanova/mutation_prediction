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

# count_dataset_path = ["/home/yulia/mnt/mutation_prediction_data/tumour_mutation_counts_dataset.part1.pickle",
#                         "/home/yulia/mnt/mutation_prediction_data/tumour_mutation_counts_dataset.part2.pickle",
#                         "/home/yulia/mnt/mutation_prediction_data/tumour_mutation_counts_dataset.part3.pickle",
#                          "/home/yulia/mnt/mutation_prediction_data/tumour_mutation_counts_dataset.part4.pickle",
#                           "/home/yulia/mnt/mutation_prediction_data/tumour_mutation_counts_dataset.part5.pickle",
#                            "/home/yulia/mnt/mutation_prediction_data/tumour_mutation_counts_dataset.part6.pickle"]

count_dataset_path = ["/Users/yulia/Documents/mutational_signatures/mutation_prediction_data/tumour_mutation_counts_dataset.small.pickle"]
model_params_file = "trained_models/model.mut_counts.lat{}.lambda_neural_net.small.pickle"

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

class NeuralNet():
    def __init__(self, params = None, layer_sizes = None, scale = 0.1, rs=npr.RandomState(0)):
        """Build a list of (weights, biases) tuples,
           one for each layer in the net.
           Either initialize weights with input params or randomly (using layer_sizes and scale)
        """
        self.layer_sizes = layer_sizes

        if params is not None:
            self.param = params
        else:
            if layer_sizes is None:
                raise Exception("Please provide the layer sizes")

            self.params = [(scale * rs.randn(m, n),   # weight matrix
                 scale * rs.randn(n))      # bias vector
                for m, n in zip(layer_sizes[:-1], layer_sizes[1:])]

    def pack_params(self):
        packed = []
        for i in range(len(self.layer_sizes)-1):
            weights = self.params[i][0].flatten()
            biases = self.params[i][1]
            packed.append(np.concatenate((weights, biases)))

        packed = np.concatenate(packed)
        return packed

    def unpack_params(self, packed):
        current = 0
        unpacked = []
        for m, n in zip(self.layer_sizes[:-1], self.layer_sizes[1:]):
            weights = packed[current:current + m*n].reshape((m,n))
            current += m*n

            biases = packed[current:current+n]
            current += n

            unpacked.append((weights, biases))
        return unpacked

    def neural_net_predict(self, inputs):
        """Implements a deep neural network for classification.
           params is a list of (weights, bias) tuples.
           inputs is an (N x D) matrix.
           returns normalized class log-probabilities."""
        for W, b in self.params:
            outputs = np.dot(inputs, W) + b
            inputs = np.tanh(outputs)
        return outputs # - logsumexp(outputs, axis=1, keepdims=True)

class tensorDecomposition():
    def __init__(self, data, data_dimensions, latent_dimension):
        self.dimensions = data_dimensions
        self.latent_dimension = latent_dimension
        self.data = data

        num_latent_params = sum([d * latent_dimension for d in dimensions.values()])
        self.params = rs.randn(num_latent_params) / 2
        self.iter = 0
        self.nn = NeuralNet(layer_sizes = [3*latent_dimension, 50, 1])

    def unpack_params(self, params):
        sample_latents_dim = self.dimensions['sample'] * self.latent_dimension
        region_latents_dim = self.dimensions['region'] * self.latent_dimension
        type_latents_dim = self.dimensions['type'] * self.latent_dimension

        sample_latents = np.reshape(params[:sample_latents_dim], (self.dimensions['sample'],latent_dimension))
        region_latents = np.reshape(params[sample_latents_dim:(sample_latents_dim+region_latents_dim)], (dimensions['region'],latent_dimension))
        type_latents = np.reshape(params[(sample_latents_dim+region_latents_dim):(sample_latents_dim+region_latents_dim+type_latents_dim)], (self.dimensions['type'],latent_dimension))
        nn_params = self.nn.unpack_params(params[(sample_latents_dim+region_latents_dim+type_latents_dim):])

        return sample_latents, region_latents, type_latents, nn_params


    def concatenate_latents(self, sample_latents, region_latents, type_latents):
        # make all combinations of vectors from the three input matrices        
        n_samples, n_latents = sample_latents.shape
        n_regions = region_latents.shape[0]
        n_types = type_latents.shape[0]

        print("hello")
        # make possible combinations across samples and regions
        samples_regions = np.concatenate((np.tile(sample_latents,(1,n_regions)).reshape(n_regions*n_samples,n_latents), np.tile(region_latents,(n_samples,1))), axis=1).reshape((n_samples,n_regions, 2 * n_latents))
        
        print(samples_regions.shape)
        # make possible combinations across samples, regions and types
        samples_regions_types =  np.concatenate((np.tile(samples_regions,(1,n_types)).reshape(n_regions*n_samples*n_types,n_latents*2), np.tile(type_latents,(n_samples*n_regions,1))), axis=1)
        print(samples_regions_types.shape)

        # simple check that everything was propagated correctly
        assert(all(samples_regions_types.reshape((n_samples,n_regions, n_types, 3 * n_latents))[0,1,2] == np.concatenate((sample_latents[0], region_latents[1], type_latents[2]))))

        return samples_regions_types

    def get_lambda(self,params):
        print("hello3")
        sample_latents, region_latents, type_latents, nn_params = self.unpack_params(params)
        n_samples, n_latents = sample_latents.shape
        n_regions = region_latents.shape[0]
        n_types = type_latents.shape[0]

        print("hello2")
        #log_lam = np.dot((region_latents * sample_latents[:, np.newaxis]), type_latents.T)
        #log_lam = self.concatenate_latents(sample_latents, region_latents, type_latents).sum(axis=1)

        self.nn.params = nn_params
        #log_lam = self.nn.neural_net_predict(self.concatenate_latents(sample_latents, region_latents, type_latents))
        log_lam = log_lam.reshape((n_samples,n_regions, n_types))

        print(log_lam)

        return log_lam

    def logprob(self, params):
        print("hello4")
        return np.sum(poisson_pmf(self.data, self.get_lambda(params))) / self.dimensions['sample'] / self.dimensions['region'] / self.dimensions['type']

    def callback(self, params):
        self.iter += 1
        print("{}: Log likelihood {:.3f}".format(self.iter, - self.objective(params)))
        
        model_params_file_tmp = "trained_models/model.mut_counts.lat{}.small.tmp.pickle".format(self.latent_dimension)
        dump_pickle(self.params, model_params_file_tmp)

    def objective(self, params):
        print("hello4")
        self.params = params
        return - self.logprob(params)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train model to predict mutation rates per region')
    parser.add_argument('--test', help='test mode: only read trained params and analyze', action="store_true")
    parser.add_argument('--latents', help='number of latent dimensions', default=100, type=int)
    parser.add_argument('-n', '--tumours', help='number of tumours to include in the set', default=None)

    args = parser.parse_args()
    test_mode = args.test
    n_tumours = args.tumours
    latent_dimension = args.latents

    rs = npr.RandomState(0)

    print("Loading dataset...")
    data, tumour_names = load_dataset(count_dataset_path)

    data = data[:10,:15,:20]
    if n_tumours is not None:
        data = data[:int(n_tumours),:,:]
    else:
        n_tumours = data.shape[0]

    print("Processing {} tumours with {} latent dimensions...".format(data.shape[0], latent_dimension))

    dimensions = {'sample': data.shape[0], 'region': data.shape[1], 'type': data.shape[2]}
    
    model_params_file = model_params_file.format(latent_dimension)

    td = tensorDecomposition(data, dimensions, latent_dimension)

    num_latent_params = sum([d * latent_dimension for d in dimensions.values()])
    
    if not test_mode:
        print("Optimizing...")
        init_params = rs.randn(num_latent_params)

        minimize(value_and_grad(td.objective), np.concatenate((td.params, td.nn.pack_params())), method='Newton-CG', 
            jac=True, callback= td.callback, options={'maxiter':1})

        dump_pickle(td.params, model_params_file)
    else:
        if not os.path.isfile(model_params_file):
            print("Model parameters not found: " + model_params_file)
            exit()

        trained_params = load_pickle(model_params_file)
        print(trained_params.shape)

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
        exit()

        sample_latents, region_latents, type_latents, nn_params = td.unpack_params(trained_params)
        print(sample_latents)
        print(type_latents)

        #compare sample and region latents
        #experiments with diff latent dimentions
        #can it overfit?

    # compute average error
    # does lambda differ across tumours, types, etc.?
    # make training and test sets
    # predict lambda for new tumour, region


