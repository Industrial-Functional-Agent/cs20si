import tensorflow as tf


# a = tf.constant(2)
# b = tf.constant(3)
#
# x = tf.add(a, b)
a = tf.constant([2, 2], name="a")
b = tf.constant([3, 6], name="b")
x = tf.add(a, b, name="add")

with tf.Session() as sess:
    writer = tf.summary.FileWriter('./graphs', sess.graph)
    print(sess.run(x))

# close the writer when you're done using it
writer.close()


t_0 = 19  # Treated as a 0-d tensor, or "scalar"
tf.zeros_like(t_0)  # ==> 0
tf.ones_like(t_0)  # ==> !

t_1 = [b"apple", b"peach", b"grape"]  # treated as a 1-d tensor, or "vector"
tf.zeros_like(t_1)  # ==> ['' '' '']

t_2 = [[True, False, False],
       [False, False, True],
       [False, True, False]]

sess = tf.Session()
print(sess.run(tf.zeros_like(t_1)))
print(sess.run(tf.zeros_like(t_2)))

my_const = tf.constant([1.0, 2.0], name="my_const")
print(tf.get_default_graph().as_graph_def())


# create variable a with scalar value
a = tf.Variable(2, name="scalar")

# create variable b with as a vector
b = tf.Variable([2, 3], name="vector")

# create variable c as a 2x2 matrix
c = tf.Variable([[0, 1], [2, 3]], name="matrix")

# create variable W as 784 x 10 tensor, filled with zeros
W = tf.Variable(tf.zeros([784, 10]))

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

init_ab = tf.variables_initializer([a, b], name="init_ab")
with tf.Session() as sess:
    sess.run(init_ab)


W = tf.Variable(tf.zeros([784, 10]))
with tf.Session() as sess:
    sess.run(W.initializer)
    print(sess.run(W))


# W is a random 700 x 100 variable object
W = tf.Variable(tf.truncated_normal([700, 10]))
with tf.Session() as sess:
    sess.run(W.initializer)
    print(W.eval())

W = tf.Variable(10)
W.assign(100)
with tf.Session() as sess:
    sess.run(W.initializer)
    print(W.eval()) # >> 10


assign_op = W.assign(100)
with tf.Session() as sess:
    sess.run(assign_op)
    print(W.eval())


# create a variable whose original value is 2
a = tf.Variable(2, name='scalar')

# assign a * 2 to a and call that op a_times_two
a_times_two = a.assign(a * 2)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    # have to initialize a, because a_times_two op depends on the value of a
    print(sess.run(a_times_two))
    print(sess.run(a_times_two))
    print(sess.run(a_times_two))

sess = tf.InteractiveSession()
a = tf.constant(5.0)
b = tf.constant(6.0)
c = a * b
# We can just use 'c.eval' without passing 'sess'

print(c.eval())
sess.close()

# create a placeholder of type float 32-bit, shape is a vector of 3 elements
a = tf.placeholder(tf.float32, shape=[3])

# create a constant of type float 32-bit, shape is a vector of 3 elements
b = tf.constant([5, 5, 5], tf.float32)

# use the placeholder as you would a constant or a variable
c = a + b

with tf.Session() as sess:
    writer = tf.summary.FileWriter('./my_graph', sess.graph)
    print(sess.run(c, feed_dict={a: [1, 1, 1]}))

writer.close()


a = tf.add(2, 5)
b = tf.multiply(a, 3)

sess = tf.Session()

replace_dict = {a: 15}

print(sess.run(b, feed_dict=replace_dict))

x = tf.Variable(10, name='x')
y = tf.Variable(20, name='y')
z = tf.add(x, y)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for _ in range(10):
        sess.run(z)

# stupid method
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for _ in range(10):
        sess.run(tf.add(x, y))


