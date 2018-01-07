import tensorflow as tf
import tempfile

def deepnn(x):
  """deepnn builds the graph for a deep net for classifying digits.
  Args:
    x: an input tensor with the dimensions (N_examples, 784), where 784 is the
    number of pixels in a standard MNIST image.
  Returns:
    A tuple (y, keep_prob). y is a tensor of shape (N_examples, 10), with values
    equal to the logits of classifying the digit into one of 10 classes (the
    digits 0-9). keep_prob is a scalar placeholder for the probability of
    dropout.
  """
  # Reshape to use within a convolutional neural net.
  # Last dimension is for "features" - there is only one here, since images are
  # grayscale -- it would be 3 for an RGB image, 4 for RGBA, etc.
  with tf.name_scope('reshape'):
    x_image = tf.reshape(x, [-1, 344, 9, 1])

  # First convolutional layer - maps one grayscale image to 32 feature maps.
  with tf.name_scope('conv1'):
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

  # Pooling layer - downsamples by 2X.
  with tf.name_scope('pool1'):
    h_pool1 = max_pool_2x2(h_conv1)

  # Second convolutional layer -- maps 32 feature maps to 64.
  with tf.name_scope('conv2'):
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

  # Second pooling layer.
  with tf.name_scope('pool2'):
    h_pool2 = max_pool_2x2(h_conv2)

  # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
  # is down to 7x7x64 feature maps -- maps this to 1024 features.
  with tf.name_scope('fc1'):
    W_fc1 = weight_variable([16512, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 16512])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

  # Dropout - controls the complexity of the model, prevents co-adaptation of
  # features.
  with tf.name_scope('dropout'):
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

  # Map the 1024 features to 10 classes, one for each digit
  with tf.name_scope('fc2'):
    W_fc2 = weight_variable([1024, 1])
    b_fc2 = bias_variable([1])

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
  return y_conv, keep_prob


def conv2d(x, W):
	"""conv2d returns a 2d convolution layer with full stride."""
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
	"""max_pool_2x2 downsamples a feature map by 2X."""
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
						strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(shape, mean = 0.0, stddev=0.1):
	"""weight_variable generates a weight variable of a given shape."""
	initial = tf.truncated_normal(shape, mean = mean, stddev=stddev)
	return tf.Variable(initial)


def bias_variable(shape, value=0.1):
	"""bias_variable generates a bias variable of a given shape."""
	initial = tf.constant(value, shape=shape)
	return tf.Variable(initial)

def unpack_gaussian(x, dim):
	return x[:,:dim], x[:,dim:]

def get_batch(data_list, batch_size, i):
	return [data[i*batch_size:(i+1)*batch_size] for data in data_list]

def save_graph():
	graph_location = tempfile.mkdtemp()
	print('Saving graph to: %s' % graph_location)
	train_writer = tf.summary.FileWriter(graph_location)
	train_writer.add_graph(tf.get_default_graph())

def init_neural_net_params(layer_sizes):
	"""Build a list of (weights, biases) tuples,
		 one for each layer in the net."""
	weights = []
	biases = []

	for i, (m, n) in enumerate(list(zip(layer_sizes[:-1], layer_sizes[1:]))):
		print((m,n))
		weights.append(weight_variable([m, n]))
		biases.append(bias_variable([n]))

	return {'weights': weights, 'biases': biases}

def neural_net(data, weights, biases):
	x = data
	for w,b in list(zip(weights, biases))[:-1]:
		x = tf.nn.relu(tf.matmul(x, w) + b)
		#x = tf.tanh(tf.matmul(x, w) + b)

	output = tf.matmul(x, weights[-1]) + biases[-1]
	return output


def correct_predictions(predictions, y_):
  predictions_binary = tf.cast(tf.less(tf.constant(0.5), predictions),tf.int64) # gaussian
  correct_prediction = tf.equal(predictions_binary, tf.cast(y_,tf.int64))
  correct_prediction = tf.cast(correct_prediction, tf.float32)
  return correct_prediction

def correct_predictions_multiclass(predictions, y_):
	correct_prediction = tf.equal(tf.argmax(predictions, axis=1),  tf.argmax(y_, axis=1))
	correct_prediction = tf.cast(correct_prediction, tf.float32)
	return correct_prediction

def mean_squared_error(truth, predicted):
  return tf.reduce_mean(tf.square(truth - predicted))
    