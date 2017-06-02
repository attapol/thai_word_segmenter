#!/usr/bin/python
# -*- coding: utf8 -*-
import codecs

import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
from data_util import iterate_seq

def main():

    n_steps = 85
    n_hidden = 20
    n_classes = 2
    pretrained_char_vector = True


    if pretrained_char_vector:
        char_dict = CharDict('char_vectors/vec.txt')
        n_vocab = char_dict.size()
        embeddings = char_dict.char_matrix
        n_input = embeddings.shape[1]
    else:
        char_dict = CharDict()
        n_vocab = char_dict.size()
        n_input = 15
        embeddings = np.random.rand(n_vocab, n_input) * 0.0001

    #data_x, data_y, data_length = load_fake_data()
    data_x, data_y, data_length = load_data('./data/train_orchid97_features.bi', char_dict, n_steps)
    dev_x, dev_y, dev_length = load_data('./data/dev_orchid97_features.bi', char_dict, n_steps)

    char_ids = tf.placeholder(tf.int32, shape=[None, n_steps], name='character_ids_input')
    labels = tf.placeholder(tf.int32, [None, n_steps], name='labels')
    sequence_lengths = tf.placeholder(tf.int32, shape=[None], name='sequence_lengths')
    mask = tf.sequence_mask(sequence_lengths)


    char_lexicon = tf.Variable(embeddings, dtype=tf.float32, trainable=True)
    parameter_summaries = []
    parameter_summaries.append(tf.summary.histogram('Char_lexicon', char_lexicon))
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
        W = tf.Variable(tf.random_uniform(minval=0,
                                          maxval= 2.0 / np.sqrt(2 * n_hidden),
                                          shape=[2 * n_hidden, n_classes]),
                        name='W')
        b = tf.Variable(tf.zeros(n_classes), name='b')

        parameter_summaries.append(tf.summary.histogram('weights', W))
        parameter_summaries.append(tf.summary.histogram('biases', b))

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


    merged_summary = tf.summary.merge([tf.summary.scalar('Loss', loss), tf.summary.scalar('Accuracy', accuracy)])
    merged_param_summary = tf.summary.merge(parameter_summaries)

    batch_size = 20

    with tf.Session() as sess:
        batch_writer = tf.summary.FileWriter("./tboard/batch", sess.graph)
        #train_writer = tf.summary.FileWriter("./tboard/train", sess.graph)
        dev_writer = tf.summary.FileWriter("./tboard/dev", sess.graph)
        init = tf.global_variables_initializer()
        sess.run(init)

        test_parse(char_dict, class_prediction, char_ids, sequence_lengths, sess, n_steps)

        step = 1
        display_step = 5
        num_epoch = 100
        for ei in range(num_epoch):
            for batch_x, batch_y, batch_length in batch_iter([data_x, data_y, data_length], batch_size, shuffle=True):
                _, l, cp, summ, param_summ = sess.run([train_op, loss, class_prediction, merged_summary, merged_param_summary],
                                      feed_dict={char_ids: batch_x, labels: batch_y, sequence_lengths: batch_length})
                batch_writer.add_summary(summ, step)
                batch_writer.add_summary(param_summ, step)

                acc = sess.run([accuracy],
                                  feed_dict={char_ids: data_x, labels: data_y, sequence_lengths: data_length})
                print (acc)

                if step % display_step == 0:
                    #_, l, cp, summ = sess.run([train_op, loss, class_prediction, merged_summary],
                                              #feed_dict={char_ids: data_x, labels: data_y, sequence_lengths: data_length})
                    #train_writer.add_summary(summ, step)
                    test_parse(char_dict, class_prediction, char_ids, sequence_lengths, sess, n_steps)
                    _, l, cp, summ = sess.run([train_op, loss, class_prediction, merged_summary],
                                              feed_dict={char_ids: dev_x, labels: dev_y, sequence_lengths: dev_length})
                    dev_writer.add_summary(summ, step)
                    print ("Iter " + str(step) + ", Minibatch Loss= " + \
                          "{:.6f}".format(l))
                step += 1
        print ('Optimization finished')


def load_fake_data():
    x = np.matrix([[2,1,3,0], [4,1,2,0], [1,2,4,0], [3,1,4,0]], dtype=int)
    y = np.matrix([[0,1,0,0], [0,1,0,0], [1,0,0,0], [0,1,0,0]], dtype=int)
    length = np.array([3, 4, 3, 3], dtype=int)
    return x, y, length

def test_parse(char_dict, class_prediction_tensor, char_ids, sequence_lengths, sess, max_seq_length):
    thai_texts = [u'อุบัติเหตุสุดหวาดเสียวผลจากพายุฝนที่ถล่มลงมาอย่างหนักในช่วงเช้า',
                  u'จอห์นเดินทางกลับสู่ประเทศไทย',
                  u'สถาบันเกอเต้จากประเทศแถบยุโรป',
                  u'พรุ่งนี้จะอยู่บ้านอาบน้ำกินนมนอน'
                  ]
    char_index_seq_list = []
    data_length = []

    for thai_text in thai_texts:
        data_length.append(len(thai_text))
        num_padding = max_seq_length - len(thai_text)
        char_index_seq = [char_dict.to_index(x) for x in thai_text]
        char_index_seq = char_index_seq + [char_dict.to_index(char_dict.UNK) for _ in range(num_padding)]
        char_index_seq_list.append(char_index_seq)

    prediction = sess.run([class_prediction_tensor],
                              feed_dict={char_ids: char_index_seq_list, sequence_lengths: data_length})[0]
    for i in range(len(thai_texts)):
        length = data_length[i]
        thai_text = thai_texts[i]
        labels = prediction[i][0:length]

        string_builder = []
        for j in range(length):
            label = labels[j]
            character = thai_text[j]
            if j != 0 and label == 1:
                string_builder.append(u'|')
            string_builder.append(character)
        print (''.join(string_builder))


class CharDict(object):

    UNK = u'__UNK__'

    def __init__(self, embedding_file=None):
        self.char_matrix = None
        if embedding_file is None:
            consonants = [chr(ord(u'ก') + i) for i in range(46)]
            vowel_set1 = [chr(ord(u'ฯ') + i) for i in range(12)]
            vowel_tone_numbers =[chr(ord(u'฿') + i) for i in range(29)]
            thai_char_list = consonants + vowel_set1 + vowel_tone_numbers
            self.thai_char_to_index = {x:i for i, x in enumerate(thai_char_list)}
            self.index_to_thai_char = {i:x for i, x in enumerate(thai_char_list)}
        else:
            self.thai_char_to_index = {}
            self.index_to_thai_char = {}
            with codecs.open(embedding_file, encoding='utf8') as f:
                vocab_size, num_units = f.readline().strip().split(' ')
                self.char_matrix = np.zeros([int(vocab_size) + 1, int(num_units)], dtype=np.float32)
                char_index = 0
                for line in f:
                    char, vector_string = line.strip().split(' ', 1)
                    vector = np.array([float(x) for x in vector_string.split(' ')])
                    self.char_matrix[char_index, :] = vector
                    self.index_to_thai_char[char_index] = char
                    self.thai_char_to_index[char] = char_index
                    char_index += 1

        num_characters = len(self.thai_char_to_index)
        self.thai_char_to_index[self.UNK] = num_characters
        self.index_to_thai_char[num_characters] = self.UNK

    def to_index(self, character):
        if character in self.thai_char_to_index:
            return self.thai_char_to_index[character]
        else:
            return self.thai_char_to_index[self.UNK]

    def to_char(self, index):
        return self.index_to_thai_char[index]

    def size(self):
        return len(self.thai_char_to_index)


def load_data(file_name, char_dict, max_seq_length=50):
    label_seq_list = []
    char_index_seq_list = []

    for seq in iterate_seq(codecs.open(file_name, encoding='utf8')):
        if len(seq) > max_seq_length:
            continue
        label_seq = [1 if label == 'B' else 0 for _, __, label in seq]
        char_index_seq = [char_dict.to_index(char) for char, _, __ in seq]

        num_padding = max_seq_length - len(seq)
        label_seq = label_seq + [0 for _ in range(num_padding)]
        label_seq_list.append(label_seq)

        char_index_seq = char_index_seq + [char_dict.to_index(char_dict.UNK) for _ in range(num_padding)]
        char_index_seq_list.append(char_index_seq)

    seq_length_list = [len(x) for x in char_index_seq_list]

    return np.matrix(char_index_seq_list, dtype=int), np.matrix(label_seq_list, dtype=int), np.array(seq_length_list, dtype=int)


def batch_iter(data, batch_size, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data_size = len(data[0])
    for i in range(1, len(data)):
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
