import tensorflow as tf


g = tf.Graph()
with g.as_default():
    x = tf.add(3, 5)

sess = tf.Session(graph=g)
print(sess.run(x))

with tf.Session(graph=g) as sess:
    print(sess.run(x))

# g = tf.get_default_graph()
