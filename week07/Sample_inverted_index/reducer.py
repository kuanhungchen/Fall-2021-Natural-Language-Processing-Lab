#!/usr/bin/env python
# -*- coding: utf-8 -*-
from itertools import groupby
from operator import itemgetter


def inverted_index_reduce(lines):
    prev_query = None
    cur_ngrams = []
    for line in lines:
        query, ngram, count = line.strip().split('\t')
        if prev_query != query:
            if prev_query is not None:
                yield prev_query, ('\t'.join([cur_ngram + " " + cur_count for (cur_ngram, cur_count) in cur_ngrams]))
            prev_query = query
            cur_ngrams = []
        cur_ngrams.append((ngram, count))
    yield prev_query, ('\t'.join([cur_ngram + " " + cur_count for (cur_ngram, cur_count) in cur_ngrams]))


if __name__ == '__main__':
    import fileinput
    for (query, ngrams) in inverted_index_reduce(fileinput.input()):
        print('\t'.join([query, ngrams]))
