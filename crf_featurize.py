import codecs
import sys

from data_util import iterate_seq

def crf_featurize(pattern_list, file_name):
    output_file_name = file_name + '.feature'
    with codecs.open(file_name, encoding='utf8') as f, \
        codecs.open(output_file_name, encoding='utf8', mode='w') as output:
        for seq in iterate_seq(f):
            seq_label_feature = featurize(pattern_list, seq)        
            for label, feature in seq_label_feature:
                output.write(u'{}\t{}\n'.format(label, '\t'.join(feature)))
            output.write(u'\n')

def read_template(template):
    patterns = []
    with open(template) as f:
        for line in f:
            if ',' in line: #conjoined feature
                pattern = [int(x) for x in line.strip().split(',')]
            else:
                pattern = [int(line.strip())]
            patterns.append(pattern)
    return patterns


def featurize(pattern_list, seq):
    label_feature_seq = []
    for i in range(len(seq)):
        features = []
        _, _, label = seq[i] 
        for pattern in pattern_list:
            conjoined_features = []
            feature_name = u't{}'.format('t'.join([str(x) for x in pattern]))
            for position in pattern:
                if (i + position) >= 0 and (i + position) < len(seq):
                    character, _, _ = seq[i+position]
                    conjoined_features.append(character)
                else:
                    conjoined_features.append(u'N')
            features.append(
                u'{}_{}'.format(feature_name, u''.join(conjoined_features)))
        label_feature_seq.append( (label, features) )
    return label_feature_seq

if __name__ == '__main__':
    template = sys.argv[1]
    file_names = sys.argv[2:]
    patterns = read_template(template)
    for file_name in file_names:
        crf_featurize(patterns, file_name)
