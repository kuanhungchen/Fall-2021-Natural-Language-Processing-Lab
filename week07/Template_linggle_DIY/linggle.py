#!/usr/bin/env python
# -*- coding: utf-8 -*-
from operator import itemgetter
from itertools import product, permutations
from functools import reduce
from collections import defaultdict
from heapq import nlargest
import re
import sys
import logging


MAX_LEN = 5
TOTAL_WORDS = set()
POS_TABLE = defaultdict(set)
POS_PATTERN = "(\(prep\.\)|\(n\.\)|\(v\.\)|\(punct\.\)|\(conj\.\)|\(adv\.\)|" \
              "\(part\.\)|\(det\.\)|\(adj\.\)|\(pron\.\)|\(num\.\)|\(intj\.\))"

def parse_ngramstr(text):
    ngram, count = text.rsplit(maxsplit=1)
    return ngram, int(count)

def parse_line(line):
    query, *ngramcounts = line.strip().split('\t')
    return query, tuple(map(parse_ngramstr, ngramcounts))

def expand_query(query):
    # TODO: write your query expansion, e.g.,
    #  "in/at afternoon" -> ["in afternoon", "at afternoon"]
    #  "listen ?to music" -> ["listen music", "listen to music"]
    def decipher(token):
        if not token:
            return [None]
        if token == '_':
            return PLACEHOLDER
        elif token == '*':
            return WILDCARD
        elif '/' in token:
            return reduce(lambda a, b: a + b,
                          [decipher(opt) for opt in token.split('/')])
        elif '$' in token:
            return partial_match(token)
        elif token[0] == '?':
            return [token[1:], None]
        elif token[-1] == '.':
            return part_of_speech(token)
        else:
            return [token.replace('_', ' ')]

    def partial_match(token):
        pattern = token.replace('$', '(.*)')
        compiled = re.compile(pattern)
        valid_words = []
        for word in TOTAL_WORDS:
            res = compiled.match(word)
            if res is not None:
                valid_words.append(res.group())
        return valid_words

    def part_of_speech(token):
        if token in POS_TABLE:
            return list(POS_TABLE[token])
        return []

    def ordering(start_idx):
        end_idx = start_idx
        orders_arr = []
        for idx in range(start_idx, len(tokens)):
            pure_token = tokens[idx].replace('{', '').replace('}', '')
            orders_arr.append(decipher(pure_token))
            if tokens[idx][-1] == '}':
                end_idx = idx; break
        return (end_idx,
                [' '.join(perm) for orders in product(*orders_arr)
                 for perm in permutations([order for order in orders if order])])

    cands_arr = list()
    tokens = query.split()
    PLACEHOLDER = ['_']
    WILDCARD = [' '.join(['_' for _ in range(l)])
                for l in range(MAX_LEN - (len(tokens) - 1) + 1)]
    idx = 0
    while idx < len(tokens):
        token = tokens[idx]
        if token[0] != '{':
            cands_arr.append(tuple(decipher(token)))
        else:
            nxt_idx, orderings = ordering(idx)
            if nxt_idx == idx:
                return []
            idx = nxt_idx
            cands_arr.append(tuple(orderings))
        idx += 1
    output_queries = []
    for cands in product(*cands_arr):
        cands = [cand for cand in cands if cand]
        if len(cands) > MAX_LEN:
            continue
        output_queries.append(' '.join(cands))
    return output_queries

def extend_query(query):
    # TODO: write your query extension,
    # e.g., can tolerate missing/unnecessary determiners
    return [query]


def load_data(lines):
    logging.info('Loading...', end='')
    # read linggle data
    linggle_table = {query: ngramcounts for query, ngramcounts in map(parse_line, lines)}
    logging.info('ready.')
    for (q, _) in linggle_table.items():
        for w in q.split():
            TOTAL_WORDS.add(w)
    return linggle_table

def load_pos(lines):
    all_pos = defaultdict(dict)
    for line in lines:
        pos, *words = line.strip().split('\t')
        pos = pos[1:-1]
        for word in words:
            if not word.strip() or word.strip() == '_':
                continue
            word = word.strip().lower()
            if pos not in all_pos[word]:
                all_pos[word][pos] = 1
            else:
                all_pos[word][pos] += 1
    for word in TOTAL_WORDS:
        lowercase_word = word.lower()
        if lowercase_word in all_pos:
            pos, _ = max(all_pos[lowercase_word].items(), key=itemgetter(1))
            POS_TABLE[pos].add(word)

def linggle(linggle_table):
    q = input('linggle> ')

    # exit execution keyword: exit()
    if q == 'exit()':
        return

    # extend and expand query
    queries = [
        simple_query
        for query in extend_query(q)
        for simple_query in expand_query(query)
    ]
    # gather results
    ngramcounts = {item for query in queries if query in linggle_table for item in linggle_table[query]}
    # output 10 most common ngrams
    ngramcounts = nlargest(10, ngramcounts, key=itemgetter(1))

    if len(ngramcounts) > 0:
        print(*(f"{count:>7,}: {ngram}" for ngram, count in ngramcounts), sep='\n')
    else:
        print(' '*8, 'no result.')
    print()

    return True


if __name__ == '__main__':
    import fileinput
    # If the readline module was loaded, then input() will use it to provide elaborate line editing and history features.
    # https://docs.python.org/3/library/functions.html#input
    import readline

    linggle_table = load_data(fileinput.FileInput(sys.argv[1]))
    if len(sys.argv) > 2:
        load_pos(fileinput.FileInput(sys.argv[2]))
    while linggle(linggle_table):
        pass
