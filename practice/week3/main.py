import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import xlrd

DATA_FILE = "../../data/fire_theft.xls"

# Step 1: read in data from the .xls file
book = xlrd.open_workbook(DATA_FILE, encoding_override="utf-8")
sheet = book.sheet_by_index(0)
data = np.asarray([sheet.row_values(i) for i in range(1, sheet.nrows)])
n_samples = sheet.nrows - 1

# Step 2: create placeholders for input X (number of fire) and label Y (number of theft)
X = tf.placeholder(tf.float32, name="X")
Y = tf.placeholder(tf.float32, name="Y")

# Step 3: create weight and bias, initialized to 0
w = tf.Variable(0.0, name="weights_1")
u = tf.Variable(0.0, name="weights_2")
b = tf.Variable(0.0, name="bias")

# Step 4: construct model to predict Y (number of theft) from the number of fire
Y_predicted = X * X * w + X * u + b
# Y_predicted = X * w + b

# Step 5: use the square error as the loss function
loss = tf.square(Y - Y_predicted, name="loss")

# Step 6: using gradient descent with learning rate of 0.01 to minimize loss
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0000001).minimize(loss)

with tf.Session() as sess:
    # Step 7: initialize the necessary variables, in this case, w and b
    sess.run(tf.global_variables_initializer())

    writer = tf.summary.FileWriter('./linear_reg', sess.graph)

    # Step 8: train the model
    for i in range(100):  # run 100 epochs
        total_loss = 0

        for x, y in data:
            # Session runs train_op to minimize loss
            _, l = sess.run([optimizer, loss], feed_dict={X: x, Y: y})
            total_loss += l
        print('Epoch {0}: {1}'.format(i, total_loss/n_samples))

    writer.close()
    # Step 9: output the values of w and b
    w_value, u_value, b_value = sess.run([w, u, b])

# plot the results
X, Y = data.T[0], data.T[1]
plt.plot(X, Y, 'bo', label='Real data')
plt.plot(X, X * X * w_value + X * u_value + b_value, 'ro', label='Predicted data')
# plt.plot(X, X * w_value + b_value, 'ro', label='Predicted data')
plt.legend()
plt.show()

'''
global_step = tf.Variable(0, trainable=False, dtype=tf.int32)
learning_rate = 0.01 * 0.99 ** tf.cast(global_step, tf.float32)

increment_step = global_step.assign_add(1)
optimizer = tf.GradientDescentOptimizer(learning_rate) # learning rate can be a tensor
'''

# create an optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)

# compute the gradients for a list of variables
grads_and_vars = optimizer.compute_gradients(loss, [w, b])

# grads_and_vars is a list of tuples (gradient, variable). Do whatever you
# need to the 'gradient' part, for example, subtract each of them by 1.
subtracted_grads_and_vars = [(gv[0] - 1.0, gv[1]) for gv in grads_and_vars]

# ask the optimizer to apply the subtracted gradients.
optimizer.apply_gradients(subtracted_grads_and_vars)

# The optimizer classes automatically compute derivatives on your graph, but creators
# of new Optimizers or expert users can call the lower-level functions.


# Huber loss
def huber_loss(labels, predictions, delta=1.0):
    residual = tf.abs(predictions - labels)
    condition = tf.less(residual, delta)
    small_res = 0.5 * tf.square(residual)
    large_res = delta * residual - 0.5 * tf.square(delta)
    return tf.select(condition, small_res, large_res)

