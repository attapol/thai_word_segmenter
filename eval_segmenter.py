import codecs
import sys

from crf_featurize import iterate_seq

from mm_segmenter import MMSegmenter
from crf_segmenter import CRFSegmenter

def main(test_data):
    mm_segmenter = MMSegmenter('lexitron.txt')
    crf_segmenter = CRFSegmenter()
    raw_gold_pairs = get_raw_gold_pairs(test_data)
    print 'MM segmenter'
    evaluate(raw_gold_pairs, mm_segmenter)
    print 'CRF segmenter'
    evaluate(raw_gold_pairs, crf_segmenter)

def evaluate(raw_gold_pairs, segmenter):
    """Evaluate the segmenter

    The evaluation is done only based on the sequence accuracy.
    TODO: Word-level precision recall f1 
    """
    sequence_accuracy = 0.0
    total = 0.0
    for raw, gold in raw_gold_pairs:
        predicted = segmenter.segment(raw)
        if predicted == gold:
            sequence_accuracy += 1
        else:
            print
            print u'Predicted : {}'.format(predicted)
            print u'Gold : {}'.format(gold)
        total += 1
    sequence_accuracy = sequence_accuracy / total
    print sequence_accuracy

    
def get_raw_gold_pairs(test_data):
    pairs = []
    for seq in iterate_seq(codecs.open(test_data, encoding='utf8')):
        gold = []
        raw = []
        for char, _, label in seq:
            raw.append(char)
            if label == 'B':
                gold.append(' ')
            gold.append(char)
        gold = ''.join(gold).strip()
        raw = ''.join(raw).strip()
        pairs.append((raw, gold))
    return pairs


if __name__ == '__main__':
    test_data = sys.argv[1]
    main(test_data)
