# -*- coding: utf8 -*-
import sys
import codecs

import numpy as np

from data_util import iterate_seq

def main(file_name):
    lengths = np.array([len(seq) for seq in iterate_seq(codecs.open(file_name, encoding='utf8'))])
    print 'Mean : {}'.format(np.mean(lengths))
    print 'Median : {}'.format(np.median(lengths))
    print 'Min : {}'.format(np.min(lengths))
    print 'Max : {}'.format(np.max(lengths))
    for pc in [1, 5, 90, 99]:
        print '{}th percentile : {}'.format(pc, np.percentile(lengths, pc))


if __name__ == '__main__':
    file_names = sys.argv[1:]
    for file_name in file_names:
        print 'Statistics for {}'.format(file_name)
        main(file_name)