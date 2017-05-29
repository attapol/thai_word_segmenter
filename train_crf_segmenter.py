import sys
import codecs

import sklearn_crfsuite
from sklearn_crfsuite import metrics

from crf_featurize import featurize, iterate_seq, read_template

def main(patterns, train_data, test_data):
    crf = train(patterns, train_data)
    labels = list(crf.classes_)
    test_fseq, test_lseq = get_data(patterns, test_data)
    y_pred = crf.predict(test_fseq)
    metrics.flat_f1_score(test_lseq, y_pred, labels=labels)


def train(patterns, train_data):
    train_fseq, train_lseq = get_data(patterns, train_data)
    crf = sklearn_crfsuite.CRF()
    print 'training'
    crf.fit(train_fseq, train_lseq)
    print 'training complete!'
    return crf

def get_data(patterns, data):
    label_seq_list = []
    feature_seq_list = []
    for seq in iterate_seq(codecs.open(data, encoding='utf8')):
        seq_label_feature = featurize(patterns, seq)
        label_seq = []
        feature_seq = []
        for label, feature in seq_label_feature:
           label_seq.append(label)
           feature_seq.append({k:True for k in feature})
        label_seq_list.append(label_seq)
        feature_seq_list.append(feature_seq)
    return feature_seq_list, label_seq_list

if __name__ == '__main__':
    template = sys.argv[1]
    train_data = sys.argv[2]
    test_data = sys.argv[3]

    patterns = read_template(template)
    main(patterns, train_data, test_data)
