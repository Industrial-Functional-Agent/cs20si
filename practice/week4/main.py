import tensorflow as tf
from process_data import process_data

"""
Implement Word2Vec Skip-Gram model.
Use Noise Contrastive Estimation, because it has nice
theoretical guarantees that its derivative tends towards the 
gradient of the softmax function.
"""

VOCAB_SIZE = 50000
BATCH_SIZE = 128
EMBED_SIZE = 128  # dimension of the word embedding vectors
SKIP_WINDOW = 1  # the context window
NUM_SAMPLED = 64  # Number of negative examples to sample.
LEARNING_RATE = 1.0
NUM_TRAIN_STEPS = 10000
SKIP_STEP = 2000  # how many steps to skip before reporting the loss


def word2vec(batch_gen):
    """
    Build the graph for word2vec model and train it
    """
    # Phase 1: Assemble the graph
    # 1. Define placeholders for input and output
    with tf.name_scope('data'):
        center_words = tf.placeholder(tf.int32, shape=[BATCH_SIZE], name='center_words')
        target_words = tf.placeholder(tf.int32, shape=[BATCH_SIZE, 1], name='target_words')

    # 2. Define the weight (in this case, embedding matrix)
    with tf.name_scope('embed'):
        embed_matrix = tf.Variable(tf.random_uniform([VOCAB_SIZE, EMBED_SIZE],
                                                     -1.0, 1.0), name='embed_matrix')

    # 3. Inference (compute the forward path of the graph)
    with tf.name_scope('loss'):
        embed = tf.nn.embedding_lookup(embed_matrix, center_words, name='embed')

        # 4. Define the loss function
        nce_weight = tf.Variable(tf.truncated_normal([VOCAB_SIZE, EMBED_SIZE],
                                                     stddev=1.0 / EMBED_SIZE ** 0.5),
                                 name='nce_weight')
        nce_bias = tf.Variable(tf.zeros([VOCAB_SIZE]), name='nce_bias')

        loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weight,
                                             biases=nce_bias,
                                             labels=target_words,
                                             inputs=embed,
                                             num_sampled=NUM_SAMPLED,
                                             num_classes=VOCAB_SIZE), name='loss')

    optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss)

    # Phase 2: Execute the computation

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        total_loss = 0.0  # we use this to calculate late average loss in the last SKIP_STEP steps
        writer = tf.summary.FileWriter('./word2vec', sess.graph)
        for index in range(NUM_TRAIN_STEPS):
            centers, targets = next(batch_gen)
            loss_batch, _ = sess.run([loss, optimizer],
                                     feed_dict={center_words: centers, target_words: targets})
            total_loss += loss_batch
            if (index + 1) % SKIP_STEP == 0:
                print('Average loss at step {}: {:5.1f}'.format(index,
                                                                total_loss / SKIP_STEP))
                total_loss = 0.0
        writer.close()


def main():
    batch_gen = process_data(VOCAB_SIZE, BATCH_SIZE, SKIP_WINDOW)
    word2vec(batch_gen)

if __name__ == '__main__':
    main()
