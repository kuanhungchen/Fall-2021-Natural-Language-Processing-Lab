import re
import sys

MAX_DIST = 5

def preprocess(text):
    # preprocess/tokenize the sentence
    text = text.lower()
    tokens = re.split('\W+', text)
    return [token for token in tokens if token.isalnum()]

def _map(text: str):
    tokens = preprocess(text)

    # [ TODO ] generate the mapper output
    # Output: "{pivot}\t{word}\t{distance}\t{count}"
    # Example:
    #   predict is  -3  1
    #   predict used    -2  1
    #   predict the -1  1
    #   predict the 1   1
    #   ...

    N = len(tokens)
    ignore_weird_char = False  # True if I want to ignore weird chars
    valid = set([c for c in '0123456789abcdefghijklmnopqrstuvwxyz_'])  # chars that I care
    for i, pivot in enumerate(tokens):
        for (word, distance) in [(tokens[j], j - i) for j in range(max(0, i - MAX_DIST), min(N, i + MAX_DIST + 1))]:
            if distance == 0:
                # zero distance means pivot equals word
                continue
            if ignore_weird_char:
                # if the flag is set, then ignore lines that contain weird chars such as greek alphabet
                if any(c not in valid for c in pivot) or any(c not in valid for c in word):
                    continue
            print("{}\t{}\t{}\t{}".format(pivot, word, distance, 1))

if __name__== "__main__" :
    for line in sys.stdin:
        _map(line)
