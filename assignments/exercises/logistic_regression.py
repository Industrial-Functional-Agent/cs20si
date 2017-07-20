import tensorflow as tf
import numpy as np


"""
There is no way to increase model capacity 
I think network makes all the outputs (probability) 0 and
make the correctness 50 percent.
"""


# make weight variable
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


# make bias variable
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# Step 1: data loading and preprocessing
famhist = np.genfromtxt('../../data/heart.csv', delimiter=',',
                        skip_header=1, dtype=str, usecols=[4])
famhist = (famhist == 'Present').astype(np.float32)

heart_data = np.genfromtxt('../../data/heart.csv', delimiter=',',
                           skip_header=1, dtype=np.float32,
                           usecols=[i for i in range(10) if i != 4])
# labeling
heart_label = heart_data[:, -1].reshape([462, 1])
heart_label = np.concatenate((heart_label, 1 - heart_label), axis=1)

# normalization
heart_data = heart_data[:, :-1]
normalized_heart_data = np.divide(heart_data, np.linalg.norm(heart_data, axis=0))

heart_data = np.concatenate((normalized_heart_data, famhist.reshape([462, 1])),
                            axis=1)

train_data = heart_data[:400]
train_label = heart_label[:400]
test_data = heart_data[400:]
test_label = heart_label[400:]

# Step 2: create placeholders for input X (number of fire) and label Y (number of theft)
X = tf.placeholder(tf.float32, name="X")
Y = tf.placeholder(tf.float32, name="Y")

# Step 3: create weight and bias
w_1 = weight_variable([9, 100])
b_1 = bias_variable([100])

keep_prob = tf.placeholder(tf.float32)

w_2 = weight_variable([100, 100])
b_2 = bias_variable([100])

w_3 = weight_variable([100, 2])
b_3 = bias_variable([2])

# Step 4: construct model to predict Y
hidden_state = tf.nn.relu(tf.matmul(X, w_1) + b_1)
hidden_state_1 = tf.nn.dropout(hidden_state, keep_prob)
hidden_state2 = tf.nn.relu(tf.matmul(hidden_state_1, w_2) + b_2)
hidden_state2_1 = tf.nn.dropout(hidden_state2, keep_prob)
logits = tf.matmul(hidden_state2_1, w_3) + b_3

entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y, name='loss')
loss = tf.reduce_mean(entropy)  # compute the mean over examples in the batch

# Step 6: using gradient descent with learning rate of 0.01 to minimize loss
optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)
init = tf.global_variables_initializer()


with tf.Session() as sess:
    # to visualize using TensorBoard
    writer = tf.summary.FileWriter('./logistic_reg', sess.graph)

    sess.run(init)
    for i in range(1000):
        total_loss = 0
        _, l = sess.run([optimizer, loss],
                        feed_dict={X: train_data, Y: train_label, keep_prob: 0.5})
        total_loss += l
        print('Average loss epoch {0}: {1}'.format(i, total_loss))
    print('Optimization Finished')

    # test the model

    preds = tf.nn.softmax(logits)
    correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))
    total_correct_preds = 0
    accuracy_batch = sess.run(accuracy,
                              feed_dict={X: test_data, Y: test_label, keep_prob: 1.0})
    total_correct_preds += accuracy_batch
    print('Accuracy {0}'.format(total_correct_preds))

    writer.close()
