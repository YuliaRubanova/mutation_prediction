import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot
import matplotlib.pyplot as plt

import matplotlib.cm as cm
import matplotlib.patches as mpatches
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns; sns.set()

def plot_elbo(elbo_list, likelihood_list, kl_divergence_list, plot_name):
	# visually compare the ELBO
	plt.figure(figsize=(12,8))

	plt.plot(np.arange(len(elbo_list)), elbo_list,
			 '-', label="ELBO", alpha=.5, c="r")
	plt.plot(np.arange(len(likelihood_list)), likelihood_list, '-',
			 label="Log likelihood", c="g")
	plt.plot(np.arange(len(kl_divergence_list)), kl_divergence_list, '-',
			 label="KL divergence", c="b")

	plt.ylim((0, max(elbo_list)))
	plt.xlim((0, len(kl_divergence_list)-0.5))
	plt.xlabel("Iteration")
	plt.ylabel("ELBO")
	plt.legend(loc='lower right')
	#plt.title("%d dimensional posterior"%D)
	plt.savefig(plot_name)

def plot_loss(loss_list, plot_name, label):
	# visually compare the ELBO
	plt.figure(figsize=(12,8))

	plt.plot(np.arange(len(loss_list)), loss_list,
			 '-', label=label, alpha=.5, c="r")

	plt.ylim((0, max(loss_list)))
	plt.xlim((0, len(loss_list)-0.5))
	plt.xlabel("Iteration")
	plt.ylabel(label)
	#plt.title("%d dimensional posterior"%D)
	plt.savefig(plot_name)

def plot_likelihood(train_likelihood, test_likelihood, plot_name):
	# visually compare the ELBO
	plt.figure(figsize=(12,8))

	plt.plot(np.arange(len(train_likelihood)), train_likelihood,
			 '-', label="Train likelihood", alpha=.5, c="r")
	plt.plot(np.arange(len(test_likelihood)), test_likelihood, '-',
			 label="Test likelihood", c="b")

	plt.ylim((0, max(max(train_likelihood), max(test_likelihood))))
	plt.xlim((0, len(test_likelihood)-0.5))
	plt.xlabel("Iteration")
	plt.ylabel("Likelihood")
	plt.legend(loc='lower right')
	#plt.title("%d dimensional posterior"%D)
	plt.savefig(plot_name)

def plot_mut_types(weights, plot_name):
	# Plot histogram over mutation types
	#print(sample_from_prior)
	# time point 1
	plt.cla()
	plt.figure(figsize=(12,8))
	his = plt.hist(list(range(96)), weights=weights, bins=list(range(96)),color="gray")
	offset = .5
	colors = ["red"]*16 + ["green"]*16 + ["blue"]*16 + ["orange"]*16 + ["cyan"]*16 + ["yellow"]*16
	plt.bar(his[1][:-1],his[0],width=1, color=colors)
	plt.xlabel('Mutation type')
	plt.ylabel('Probability')

	patches = [mpatches.Patch(color='red', label='C>A')] + \
				[mpatches.Patch(color='green', label='C>G')] +  \
				[mpatches.Patch(color='blue', label='C>T')] + \
				[mpatches.Patch(color='orange', label='T>A')] + \
				[mpatches.Patch(color='cyan', label='T>C')] + \
				[mpatches.Patch(color='yellow', label='T>G')]

	plt.legend(handles=patches, prop={'size':10})
	plt.savefig(plot_name)
	plt.close()


def plot_boxplot(data, plot_name, feature_names = None, text_=""):
	plt.cla()
	fig = plt.figure(figsize=(12,15))
	ax = fig.add_subplot(111)

	ax.boxplot(data, labels = feature_names)
	plt.xticks(rotation=90)

	if (text_ != ""):
		ax.text(1, max(np.max(np.array(data))) - 0.1, text_, fontsize=15)
	
	plt.tight_layout()
	plt.savefig(plot_name)
	plt.close()

def plot_bars(data, plot_name, feature_names = None, ylabel = "", text_="", ylim = None):
	x = np.arange(len(data))

	plt.cla()
	fig = plt.figure(figsize=(12,15))
	ax = fig.add_subplot(111)

	plt.bar(x, data, color='g')
	plt.xticks(x, feature_names, rotation=90)
	plt.ylabel(ylabel)
	if ylim is not None:
		plt.ylim(ylim)

	if (text_ != ""):
		if ylim is not None:
			y_pos = ylim[1] - 0.02
		else:
			y_pos =  np.max(np.array(data)) + 0.01

		ax.text(1, y_pos, text_, fontsize=15)
		
	plt.tight_layout()
	plt.savefig(plot_name)
	plt.close()

def plot_exposures(weights, plot_name):
	# Plot histogram of signature exposures
	plt.cla()
	plt.figure(figsize=(12,8))
	plt.hist(list(range(30)), weights=weights, bins=list(range(30)),color="gray")
	plt.xlabel('Signatures')
	plt.ylabel('Signature exposures')
	plt.xticks(np.arange(0.5,31,1), list(range(1,31)))
	plt.savefig(plot_name)
	plt.close()

def plot_tsne(data, labels, label_descr, plot_name):
	plt.figure(figsize=(12,12))
	tsne = TSNE(n_components=2, random_state=0, perplexity=30.0)
	tsne_transformed = tsne.fit_transform(data) 

	max_type = int(max(labels))

	colors = cm.jet(np.linspace(0, 1, max_type))
	#print(colors[np.floor(training_annot[:,0])])

	plt.scatter(tsne_transformed[:,0], tsne_transformed[:,1], )

	for x, y, i in zip(tsne_transformed[:,0], tsne_transformed[:,1], list(range(tsne_transformed.shape[0]))):
		plt.scatter(x, y, color=colors[int(labels[i])-1])

	patches = []
	for c, t in zip(colors, label_descr):
		patches += [mpatches.Patch(color=c, label=t)]

	plt.legend(handles=patches, prop={'size':10})
	plt.savefig(plot_name)


def plot_types_over_time(data, ylabel, vmax = None, plot_name = "tmp.pdf"):
	# assuming the following order of dimensions:
	# 0) time steps 1) tumours 2) types/features

	if len(data.shape) != 3:
		raise Exception("plot_types_over_time: Expected three-dimensional array")

	n_tumours = data.shape[1]

	plt.cla()
	fig = plt.figure()
	fig, ax_list = plt.subplots(n_tumours, 1)
	
	for i, tum in enumerate(list(np.transpose(data, (1,0,2)))):
		ax = ax_list[i]
		sns.heatmap(np.transpose(tum), cmap="YlGnBu", xticklabels=2, ax = ax, vmax = vmax)
		ax.set_title('Tumour ' + str(i))
		ax.set_xlabel('Time')
		ax.set_ylabel(ylabel)

	fig.set_size_inches((12, 5 * n_tumours))
	plt.tight_layout()
	fig.savefig(plot_name, format="pdf")
	plt.close()
