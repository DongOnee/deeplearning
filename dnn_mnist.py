'''
DNN to classify MNIST handwritten digits
'''

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("./data/mnist", one_hot=True)


# Parameters
learning_rate = 0.0025
n_epochs = 70
batch_size = 50
display_step = 1

# Network Parameters
n_hidden_1 = 1024 # 1st layer number of features
n_hidden_2 = 512 # 2nd layer number of features
n_input = 784 # MNIST data input (img shape: 28*28)
n_classes = 10 # MNIST total classes (0-9 digits)

# tf Graph input
X = tf.placeholder(tf.float32, [batch_size, n_input])
Y = tf.placeholder(tf.float32, [batch_size, n_classes])


# Create model
def multilayer_perceptron(x, weights, biases):
	list_wei = list(weights.keys())
	list_biases = list(biases.keys())

	cur = x
	for i in range(0, len(list_wei)-1):
		cur = tf.add(tf.matmul(cur, weights[list_wei[i]]), biases[list_biases[i]])
		cur = tf.nn.relu(cur)
	# Output layer with linear activation
	out_layer = tf.matmul(cur, weights['out']) + biases['out']
	return out_layer

# Store layers weight & bias
weights = {
	'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
	'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
	'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}

biases = {
	'b1': tf.Variable(tf.random_normal([1])),
	'b2': tf.Variable(tf.random_normal([1])),
	'out': tf.Variable(tf.random_normal([1]))
}

# Construct model
pred = multilayer_perceptron(X, weights, biases)

# Define loss and optimizer
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=Y))

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
#optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(loss)


# Launch the graph
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())

	# Training 
	for i in range(n_epochs):
		total_loss = 0.
		n_batches = int(mnist.train.num_examples/batch_size)
		# Loop over all batches
		for j in range(n_batches):
				X_batch, Y_batch = mnist.train.next_batch(batch_size)
			# Run optimization op (backprop) and cost op (to get loss value)
				_, l = sess.run([optimizer, loss], feed_dict={X: X_batch, Y: Y_batch})
				# Compute average loss
				total_loss += l
			# Display logs per epoch step
		print('Average loss epoch {0}: {1}'.format(i, total_loss/n_batches))

	print("Optimization Finished!")


	correct_preds = tf.equal(tf.argmax(pred, axis=1), tf.argmax(Y, axis=1))
	accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))

	n_batches = int(mnist.test.num_examples/batch_size)
	total_correct_preds = 0

	for i in range(n_batches):
		X_batch, Y_batch = mnist.test.next_batch(batch_size)
		accuracy_batch = sess.run(accuracy, feed_dict={X: X_batch, Y:Y_batch}) 
		total_correct_preds += accuracy_batch   

	print('Accuracy {0}'.format(total_correct_preds/mnist.test.num_examples))
