#!/usr/bin/env python3

import sys
import traceback
import argparse
from markov import MarkovClassifier
from constants import SENTIMENT
from corpus import CorpusReader

################ CLI App ##################
def main():
    parser = argparse.ArgumentParser(prog="tester", description="test the classifier")

    parser.add_argument('--file', '-f', dest='file',
                        type=str, nargs='?', required=True,
                        help='load trained model from this file')

    parser.add_argument('--pos', '-p', dest='pos',
                        type=str, nargs='?', required=True,
                        help='test with these positive reviews')

    parser.add_argument('--neg', '-n', dest='neg',
                        type=str, nargs='?', required=True,
                        help='test with these negative reviews')

    args = parser.parse_args()

    if not args.file:
        print("no load file given")
        return 1
    if not args.pos:
        print("no pos file given")
        return 1
    if not args.neg:
        print("no neg file given")
        return 1

    try:
        markov_classifier = MarkovClassifier.loadFromFile(args.file)
    except Exception as e:
        print("Error loading Markov Classifier")
        print("%s" % (e))
        traceback.print_exc()
        return 1

    results = {
            SENTIMENT.POSITIVE: {
                SENTIMENT.POSITIVE: 0,
                SENTIMENT.NEUTRAL: 0,
                SENTIMENT.NEGATIVE: 0,
            },
            SENTIMENT.NEGATIVE: {
                SENTIMENT.POSITIVE: 0,
                SENTIMENT.NEUTRAL: 0,
                SENTIMENT.NEGATIVE: 0,
            },
    }

    print("testing positive reviews...")
    pos_counter = 0
    misses = {
        'posRows': 0,
        'posCols': 0,
        'posTrans': 0,
        'negRows': 0,
        'negCols': 0,
        'negTrans': 0,
    }

    reader = CorpusReader(args.pos)
    for review in reader.reviews():
        try:
            debugInfo = {}
            sentiment = markov_classifier.classify(review, debug=False, debugInfo=debugInfo)
            results[SENTIMENT.POSITIVE][sentiment] += 1

            misses['posRows'] += debugInfo['pos']['totalRowMisses']
            misses['posCols'] += debugInfo['pos']['totalColMisses']
            misses['posTrans'] += debugInfo['pos']['totalTransMisses']
            misses['negRows'] += debugInfo['neg']['totalRowMisses']
            misses['negCols'] += debugInfo['neg']['totalColMisses']
            misses['negTrans'] += debugInfo['neg']['totalTransMisses']

            #if debugInfo['pos']['totalColMisses'] > 0:
            #    markov_classifier.classify(review, debug=True)

            pos_counter += 1
            #print(pos_counter)
        except Exception as e:
            print("Error while classifying")
            print("%s" % (e))
            traceback.print_exc()
            return 1
    print("done. %d pos reviews classified" % pos_counter)
    print(misses)

    print("testing negative reviews...")

    neg_counter = 0
    misses = {
        'posRows': 0,
        'posCols': 0,
        'posTrans': 0,
        'negRows': 0,
        'negCols': 0,
        'negTrans': 0,
    }

    reader = CorpusReader(args.neg)
    for review in reader.reviews():
        try:
            debugInfo = {}
            sentiment = markov_classifier.classify(review, debug=False, debugInfo=debugInfo)
            results[SENTIMENT.NEGATIVE][sentiment] += 1

            misses['posRows'] += debugInfo['pos']['totalRowMisses']
            misses['posCols'] += debugInfo['pos']['totalColMisses']
            misses['posTrans'] += debugInfo['pos']['totalTransMisses']
            misses['negRows'] += debugInfo['neg']['totalRowMisses']
            misses['negCols'] += debugInfo['neg']['totalColMisses']
            misses['negTrans'] += debugInfo['neg']['totalTransMisses']

            #if debugInfo['neg']['totalColMisses'] > 0:
            #    markov_classifier.classify(review, debug=True)

            neg_counter += 1
            #print(neg_counter)
        except Exception as e:
            print("Error while classifying")
            print("%s" % (e))
            traceback.print_exc()
            return 1
    print("done. %d neg reviews classified" % neg_counter)
    print(misses)

    total_counter = pos_counter + neg_counter

    print("done. In total %d reviews classified" % total_counter)
    print("---------------------------")
    print("RESULTS");
    print("---------------------------")
    print("")
    print("reviews:    |  POS  |  NEG  |")
    print("classified: +-------+-------+")
    print("      POS   | %4d  | %4d  | %4d" % (results[SENTIMENT.POSITIVE][SENTIMENT.POSITIVE],results[SENTIMENT.NEGATIVE][SENTIMENT.POSITIVE],
                                               results[SENTIMENT.POSITIVE][SENTIMENT.POSITIVE]+results[SENTIMENT.NEGATIVE][SENTIMENT.POSITIVE]))
    print("        -   | %4d  | %4d  | %4d" % (results[SENTIMENT.POSITIVE][SENTIMENT.NEUTRAL], results[SENTIMENT.NEGATIVE][SENTIMENT.NEUTRAL],
                                               results[SENTIMENT.POSITIVE][SENTIMENT.NEUTRAL] +results[SENTIMENT.NEGATIVE][SENTIMENT.NEUTRAL]))
    print("      NEG   | %4d  | %4d  | %4d" % (results[SENTIMENT.POSITIVE][SENTIMENT.NEGATIVE],results[SENTIMENT.NEGATIVE][SENTIMENT.NEGATIVE],
                                               results[SENTIMENT.POSITIVE][SENTIMENT.NEGATIVE]+results[SENTIMENT.NEGATIVE][SENTIMENT.NEGATIVE]))
    print("            +-------+-------+--------")
    print("      total | %4d  | %4d  | %4d" % (pos_counter, neg_counter, total_counter))
    print("")
    print("reviews:    |  POS | NEG  |")
    print("classified: +------+------+")
    print("      POS   | %.2f | %.2f | %.2f" % (results[SENTIMENT.POSITIVE][SENTIMENT.POSITIVE]/total_counter,results[SENTIMENT.NEGATIVE][SENTIMENT.POSITIVE]/total_counter,
                                                results[SENTIMENT.POSITIVE][SENTIMENT.POSITIVE]/total_counter+results[SENTIMENT.NEGATIVE][SENTIMENT.POSITIVE]/total_counter))
    print("        -   | %.2f | %.2f | %.2f" % (results[SENTIMENT.POSITIVE][SENTIMENT.NEUTRAL]/total_counter, results[SENTIMENT.NEGATIVE][SENTIMENT.NEUTRAL]/total_counter,
                                                results[SENTIMENT.POSITIVE][SENTIMENT.NEUTRAL]/total_counter +results[SENTIMENT.NEGATIVE][SENTIMENT.NEUTRAL]/total_counter))
    print("      NEG   | %.2f | %.2f | %.2f" % (results[SENTIMENT.POSITIVE][SENTIMENT.NEGATIVE]/total_counter,results[SENTIMENT.NEGATIVE][SENTIMENT.NEGATIVE]/total_counter,
                                                results[SENTIMENT.POSITIVE][SENTIMENT.NEGATIVE]/total_counter+results[SENTIMENT.NEGATIVE][SENTIMENT.NEGATIVE]/total_counter))
    print("            +------+------+--------")
    print("            | %.2f | %.2f | %.2f" % (pos_counter/total_counter, neg_counter/total_counter, total_counter/total_counter))
    print("")
    return 0

if __name__ == '__main__':
    sys.exit(main())


