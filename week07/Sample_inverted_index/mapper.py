#!/usr/bin/env python
# -*- coding: utf-8 -*-
import itertools


def make_query(s):
    """Given a string, return all possible queries of it.
    """
    words = s.split()
    if len(words) == 1:
        subsets = [words[0], '_']
    else:
        A = [(word, '_') for word in words]
        subsets = [' '.join(subset) for subset in itertools.product(*A)]
    for subset in subsets:
        yield subset


def inverted_index_map(lines):
    for line in lines:
        ngram, count = line.strip().split('\t')
        for possible_query in make_query(ngram):
            yield possible_query, ngram, count


if __name__ == '__main__':
    import fileinput
    for (query, ngram, count) in inverted_index_map(fileinput.input()):
        print('\t'.join((query, ngram, count)))
