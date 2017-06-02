# -*- coding: utf8 -*-

consonants = [chr(ord(u'ก') + i) for i in range(46)]
vowel_set1 = [chr(ord(u'ฯ') + i) for i in range(12)]
vowel_tone_numbers =[chr(ord(u'฿') + i) for i in range(29)]
thai_char_list = consonants + vowel_set1 + vowel_tone_numbers
THAI_CHAR_SET = set(thai_char_list)


def is_thai_char(character):
    return character in THAI_CHAR_SET


def iterate_seq(f):
    seq = []
    for line in f:
        line = line.strip()
        if line == '' and len(seq) > 0:
            yield seq
            seq = []
        else:
            character, char_type, label = line.split(' ')
            seq.append((character, char_type, label))

