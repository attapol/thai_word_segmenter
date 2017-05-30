# -*- coding: utf8 -*-
import codecs
import sys

from data_util import is_thai_char

def main(file_name):
    for line in codecs.open(file_name, encoding='utf8'):
        line = line.strip()
        if line == '':
            if not skipped:
                print
            skipped = True
        else:
            character, char_type, label = line.split(' ')
            if is_thai_char(character):
                print line.strip()
                skipped = False
            else:
                if not skipped:
                    print
                skipped = True

if __name__ == '__main__':
    file_name = sys.argv[1]
    main(file_name)