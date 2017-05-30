#!/usr/bin/python
# -*- coding: utf8 -*-
import pygtrie 
import codecs

class MMSegmenter(object):

    def __init__(self, dict_file):
        self.word_trie = pygtrie.CharTrie()
        for word in codecs.open(dict_file, encoding='utf8'):
            self.word_trie[word.strip()] = True

    def segment(self, text):
        tokens = []
        start_pos = 0
        while start_pos < len(text):
            char_index = self._get_token(text, start_pos)
            if char_index == -1:
                char_index = self._skip_oov(text, start_pos+1)
            tokens.append(text[start_pos:char_index])
            start_pos = char_index
        return ' '.join(tokens)

    def _get_token(self, text, start_pos):
        cur_pos = start_pos
        longest_char_index = -1
        while cur_pos <= len(text):
            token_so_far = text[start_pos:cur_pos]
            if self.word_trie.has_key(token_so_far):
                longest_char_index = cur_pos
            elif not self.word_trie.has_subtrie(token_so_far):
                return longest_char_index
            cur_pos += 1
        return longest_char_index

    def _skip_oov(self, text, start_pos):
        cur_pos = start_pos
        while cur_pos < len(text):
            longest_char_index = self._get_token(text, cur_pos)
            if longest_char_index != -1:
                return cur_pos
            cur_pos += 1
        return cur_pos


if __name__ == '__main__':
    segmenter = MMSegmenter('test_lexicon.txt')
    print segmenter.word_trie.has_key(u'มานะ')
    print segmenter.segment(u'มานะนะ')
    print segmenter.segment(u'มามานะ')
    print segmenter.segment(u'เออมาร้ายนะ')
    print segmenter.segment(u'มาร้ายนะ')
    print segmenter.segment(u'มาร้ายร้ายนะเออ')
