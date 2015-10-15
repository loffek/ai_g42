#!/usr/bin/env python3

import sys
import traceback
import argparse
from markov import MarkovClassifier

################ CLI App ##################
def main():
    parser = argparse.ArgumentParser(prog="trainer", description="trains a movie review classifier on corpora")

    parser.add_argument('--order','-k',metavar='int', dest='order',
                        type=int, nargs='?', required=True, const='0',
                        help='order of the markov model. default: 0')

    parser.add_argument('--smoothing','-s', dest='smoothing',
                        type=str, nargs='?', required=True, choices=['laplace', 'backoff', 'sgts'],
                        help='smoothing technique')

    parser.add_argument('--file', '-f', dest='file',
                        type=str, nargs='?', required=True,
                        help='save trained model to this file')

    parser.add_argument('--pos', '-p', dest='pos',
                        type=str, nargs='?', required=True,
                        help='traing corpus with positive reviews')

    parser.add_argument('--neg', '-n', dest='neg',
                        type=str, nargs='?', required=True,
                        help='traing corpus with negative reviews')

    args = parser.parse_args()

    if not args.file:
        print("no save file given")
        return 1

    if not args.pos:
        print("no positive corpus given")
        return 1

    if not args.neg:
        print("no negative corpus given")
        return 1

    markov_classifier = MarkovClassifier(order=args.order, smoothing=args.smoothing)
    try:
        markov_classifier.trainOnCorpora(posfile=args.pos, negfile=args.neg)
    except Exception as e:
        print("Error training Markov Classifier")
        print("%s" % (e))
        traceback.print_exc()
        return 1
    try:
        markov_classifier.saveToFile(args.file)
    except Exception as e:
        print("Error saving Markov Classifier")
        print("%s" % (e))
        traceback.print_exc()
        return 1

    return 0

if __name__ == '__main__':
    sys.exit(main())

