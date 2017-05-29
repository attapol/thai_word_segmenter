import cPickle
from crf_featurize import featurize, iterate_seq, read_template

class CRFSegmenter(object):

    def __init__(self):
        self.model = cPickle.load(open('crf.model'))
        self.patterns = read_template('template.txt')

    def segment(self, text):
        seq = [(x, 'dummy', 'dummy') for x in text]
        label_feature_seq = featurize(self.patterns, seq)
        feature_seq = []
        for _, feature in label_feature_seq:
           feature_seq.append({k:True for k in feature})
        
        predicted = self.model.predict([feature_seq])[0]
        to_print = u''
        for pred, character in zip(predicted, text):
            if pred == 'B':
                to_print += u' '
            to_print += character
        return to_print.strip()
