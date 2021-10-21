import sys

MAX_DIST = 5

if __name__== "__main__" :

    prev_pivot = None
    prev_word = None
    prev_total = 0
    prev_count = [0 for _ in range(10)]

    for line in sys.stdin:
        line = line.strip()
        if not line: continue

        # [ TODO ] collect the output from shuffler and generate reducer output
        # Now you need to calculate the skip-gram frequency with its distance information.
        # Input:
        #   "{pivot}\t{word}\t{distance}\t{count}"
        # Output:
        #   "{pivot}\t{word}\t{total_freq}\t{-5}\t{-4}\t{-3}\t{-2}\t{-1}\t{1}\t{2}\t{3}\t{4}\t{5}"
        # Example:
        #   See the sample output file given by TA.
        # Steps:
        #   1) Parse the input from shuffler
        #   2) Check if this is the same skipgram
        #   3) If so, add the frequency according to its distance
        #   4) If not, output the previous skipgram data

        pivot, word, distance, count = line.split('\t')
        distance = int(distance)
        count = int(count)
        if prev_pivot != pivot or prev_word != word:
            # when previous pivot(word) not equals current pivot(word), print last line
            # results and reset counters
            if prev_pivot is not None:
                print("{}\t{}\t{}\t{}".format(str(prev_pivot), str(prev_word), str(prev_total), "\t".join([str(c) for c in prev_count])))
            prev_pivot = pivot
            prev_word = word
            prev_total = 0
            prev_count = [0 for _ in range(10)]
        prev_total += 1
        index = distance + 5 if distance < 0 else distance + 4  # map (-5...5) to (0...9)
        prev_count[index] += count
    print("{}\t{}\t{}\t{}".format(str(prev_pivot), str(prev_word), str(prev_total), "\t".join([str(c) for c in prev_count])))
