#!/usr/bin/python
# -*- coding: utf8 -*-
import codecs

import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
from crf_featurize import iterate_seq

def main():

    n_steps = 200
    n_input = 15
    n_hidden = 50
    n_classes = 2

    #data_x, data_y, data_length = load_fake_data()
    char_dict = CharDict()
    data_x, data_y, data_length = load_data('./train_orchid97_features.bi', char_dict, n_steps)
    dev_x, dev_y, dev_length = load_data('./dev_orchid97_features.bi', char_dict, n_steps)

    char_ids = tf.placeholder(tf.int32, shape=[None, n_steps], name='character_ids_input')
    labels = tf.placeholder(tf.int32, [None, n_steps], name='labels')
    sequence_lengths = tf.placeholder(tf.int32, shape=[None], name='sequence_lengths')
    mask = tf.sequence_mask(sequence_lengths)

    #n_vocab = 10
    n_vocab = char_dict.size()
    embeddings = [] # load the pretrained character embeddings or create new here
    embeddings = np.random.rand(n_vocab, n_input) * 0.0001
    char_lexicon = tf.Variable(embeddings, dtype=tf.float32, trainable=True)
    tf.summary.histogram('Char_lexicon', char_lexicon)
    input_embeddings = tf.nn.embedding_lookup(char_lexicon, char_ids, name='input_embedding')
    input_embeddings = tf.reshape(input_embeddings, shape=[-1, n_steps, n_input])


    lstm_forward_cell = rnn.BasicLSTMCell(n_hidden)
    lstm_backward_cell = rnn.BasicLSTMCell(n_hidden)

    # batch size x time x 2*n_hidden
    bilstm_outputs, _ = tf.nn.bidirectional_dynamic_rnn(
            lstm_forward_cell,
            lstm_backward_cell,
            input_embeddings,
            dtype=tf.float32,
            sequence_length=sequence_lengths,
            scope='BiLSTM')

    with tf.name_scope('tagging'):
        W = tf.Variable(tf.random_uniform(minval=-1, maxval=1, shape=[2 * n_hidden, n_classes]), name='W')
        b = tf.Variable(tf.zeros(n_classes), name='b')

        tf.summary.histogram('weights', W)
        tf.summary.histogram('biases', b)


        #ntime_steps = tf.shape(bilstm_outputs)[1]
        # (batch size * time) x (2*n_hidden)
        feature_flat = tf.reshape(bilstm_outputs, [-1, 2 * n_hidden])

        # (batch size * time) x n_classes
        pred = tf.matmul(feature_flat, W) + b
        scores = tf.reshape(pred, [-1, n_steps, n_classes])



    with tf.name_scope('objective'):
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=scores, labels=labels)
        losses = tf.boolean_mask(losses, mask)

        #losses, transition_params = tf.contrib.crf.crf_log_likelihood(
        #    scores, labels, sequence_lengths)
        #losses = -losses

        loss = tf.reduce_mean(losses)

    with tf.name_scope('training'):
        lr = 0.1
        optimizer = tf.train.AdagradOptimizer(lr)
        train_op = optimizer.minimize(loss)

    with tf.name_scope('accuracy'):
        class_prediction = tf.cast(tf.arg_max(scores, dimension=2, name='label_prediction'), tf.int32)
        correct_prediction = tf.cast(tf.equal(class_prediction, labels), tf.float32)
        mask = tf.sequence_mask(sequence_lengths)
        masked_correct_prediction = tf.boolean_mask(correct_prediction, mask)
        accuracy = tf.reduce_mean(masked_correct_prediction)

    tf.summary.scalar('Loss', loss)
    tf.summary.scalar('Accuracy', accuracy)
    merged_summary = tf.summary.merge_all()

    batch_size = 2

    with tf.Session() as sess:
        batch_writer = tf.summary.FileWriter("./tboard/batch", sess.graph)
        train_writer = tf.summary.FileWriter("./tboard/train", sess.graph)
        dev_writer = tf.summary.FileWriter("./tboard/dev", sess.graph)
        init = tf.global_variables_initializer()
        sess.run(init)

        step = 1
        display_step = 1
        num_epoch = 100
        for ei in xrange(num_epoch):
            for batch_x, batch_y, batch_length in batch_iter([data_x, data_y, data_length], batch_size, shuffle=True):
                _, l, cp, summ = sess.run([train_op, loss, class_prediction, merged_summary],
                                      feed_dict={char_ids: batch_x, labels: batch_y, sequence_lengths: batch_length})
                batch_writer.add_summary(summ, step)

                acc = sess.run([accuracy],
                                  feed_dict={char_ids: data_x, labels: data_y, sequence_lengths: data_length})
                print acc

                if step % display_step == 0:
                    _, l, cp, summ = sess.run([train_op, loss, class_prediction, merged_summary],
                                              feed_dict={char_ids: data_x, labels: data_y, sequence_lengths: data_length})
                    train_writer.add_summary(summ, step)
                    _, l, cp, summ = sess.run([train_op, loss, class_prediction, merged_summary],
                                              feed_dict={char_ids: dev_x, labels: dev_y, sequence_lengths: dev_length})
                    dev_writer.add_summary(summ, step)
                    print "Iter " + str(step) + ", Minibatch Loss= " + \
                          "{:.6f}".format(l)
                step += 1
        print 'Optimization finished'


def load_fake_data():
    x = np.matrix([[2,1,3,0], [4,1,2,0], [1,2,4,0], [3,1,4,0]], dtype=int)
    y = np.matrix([[0,1,0,0], [0,1,0,0], [1,0,0,0], [0,1,0,0]], dtype=int)
    length = np.array([3, 4, 3, 3], dtype=int)
    return x, y, length

class CharDict(object):

    PADDING = '__PADDING_CHAR__'

    def __init__(self):
        consonants = [unichr(ord(u'ก') + i) for i in xrange(46)]
        vowel_set1 = [unichr(ord(u'ฯ') + i) for i in xrange(12)]
        vowel_tone_numbers =[unichr(ord(u'฿') + i) for i in xrange(29)]
        thai_char_list = consonants + vowel_set1 + vowel_tone_numbers
        self.thai_char_to_index = {x:i for i, x in enumerate(thai_char_list)}

    def to_index(self, character):
        if character in self.thai_char_to_index:
            return self.thai_char_to_index[character]
        elif character == self.PADDING:
            return len(self.thai_char_to_index)
        else:
            return len(self.thai_char_to_index) + 1

    def size(self):
        return len(self.thai_char_to_index) + 2


def load_data(file_name, char_dict, max_seq_length=50):
    label_seq_list = []
    char_index_seq_list = []

    for seq in iterate_seq(codecs.open(file_name, encoding='utf8')):
        assert len(seq) < max_seq_length
        label_seq = [1 if label == 'B' else 0 for _, __, label in seq]
        char_index_seq = [char_dict.to_index(char) for char, _, __ in seq]

        num_padding = max_seq_length - len(seq)
        label_seq = label_seq + [0 for _ in range(num_padding)]
        label_seq_list.append(label_seq)

        char_index_seq = char_index_seq + [char_dict.to_index(char_dict.PADDING) for _ in range(num_padding)]
        char_index_seq_list.append(char_index_seq)

    seq_length_list = [len(x) for x in char_index_seq_list]

    return np.matrix(char_index_seq_list, dtype=int), np.matrix(label_seq_list, dtype=int), np.array(seq_length_list, dtype=int)


def batch_iter(data, batch_size, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data_size = len(data[0])
    for i in xrange(1, len(data)):
        assert len(data[i]) == data_size

    num_batches = int((data_size - 1) / batch_size) + 1

    for bi in range(num_batches):
        # Shuffle the data at each epoch
        if shuffle:
            shuffled_indices = np.random.permutation(np.arange(data_size))
        else:
            shuffled_indices = np.arange(data_size)

        start_index = bi * batch_size
        end_index = min((bi + 1) * batch_size, data_size)

        indices = shuffled_indices[start_index:end_index]
        yield (d[indices] for d in data)


if __name__ == '__main__':
    main()
