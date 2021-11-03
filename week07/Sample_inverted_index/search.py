#!/usr/bin/env python
# -*- coding: utf-8 -*-


def search(inverted_index):
    q = input(' search> ')

    # exit execution keyword: exit()
    if q == 'exit()':
        return

    res = inverted_index.get(q)

    if res:
        print(f"found in line {', '.join(res)}.")
    else:
        print('no result.')
    print()

    return True


if __name__ == '__main__':
    import fileinput
    # If the readline module was loaded, then input() will use it to provide elaborate line editing and history features.
    # https://docs.python.org/3/library/functions.html#input
    import readline

    inverted_index = {keyword: line_numbers
                      for keyword, *line_numbers in map(str.split, fileinput.input())}
    while search(inverted_index):
        pass
