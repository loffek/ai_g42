#!/usr/bin/env python3

import sys
import traceback
import argparse
from markov import MarkovClassifier

################ CLI App ##################
def main():
    parser = argparse.ArgumentParser(prog="classifier", description="classifies a movie review")

    parser.add_argument('--file', '-f', dest='file',
                        type=str, nargs='?', required=True,
                        help='load trained model from this file')

    args = parser.parse_args()

    if not args.file:
        print("no load file given")
        return 1

    try:
        markov_classifier = MarkovClassifier.loadFromFile(args.file)
    except Exception as e:
        print("Error loading Markov Classifier")
        print("%s" % (e))
        traceback.print_exc()
        return 1

    while(True):
        review = sys.stdin.readline()
        if not review:
            break;

        try:
            sentiment = markov_classifier.classify(review, debug=True)
            print(sentiment.name)
        except Exception as e:
            print("Error while classifying")
            print("%s" % (e))
            traceback.print_exc()
            return 1

    return 0

if __name__ == '__main__':
    sys.exit(main())

