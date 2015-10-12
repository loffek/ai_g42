# -*- coding: utf-8 -*-
# Natural Language Toolkit: IBM Model 5
#
# Copyright (C) 2001-2015 NLTK Project
# Author: Tah Wei Hoon <hoon.tw@gmail.com>
# URL: <http://nltk.org/>
# For license information, see LICENSE.TXT

"""
Translation model that keeps track of vacant positions in the target
sentence to decide where to place translated words.

Translation can be viewed as a process where each word in the source
sentence is stepped through sequentially, generating translated words
for each source word. The target sentence can be viewed as being made
up of ``m`` empty slots initially, which gradually fill up as generated
words are placed in them.

Models 3 and 4 use distortion probabilities to decide how to place
translated words. For simplicity, these models ignore the history of
which slots have already been occupied with translated words.
Consider the placement of the last translated word: there is only one
empty slot left in the target sentence, so the distortion probability
should be 1.0 for that position and 0.0 everywhere else. However, the
distortion probabilities for Models 3 and 4 are set up such that all
positions are under consideration.

IBM Model 5 fixes this deficiency by accounting for occupied slots
during translation. It introduces the vacancy function v(j), the number
of vacancies up to, and including, position j in the target sentence.

Terminology:
Maximum vacancy:
    The number of valid slots that a word can be placed in.
    This is not necessarily the same as the number of vacant slots.
    For example, if a tablet contains more than one word, the head word
    cannot be placed at the last vacant slot because there will be no
    space for the other words in the tablet. The number of valid slots
    has to take into account the length of the tablet.
    Non-head words cannot be placed before the head word, so vacancies
    to the left of the head word are ignored.
Vacancy difference:
    For a head word: (v(j) - v(center of previous cept))
    Can be positive or negative.
    For a non-head word: (v(j) - v(position of previously placed word))
    Always positive, because successive words in a tablet are assumed to
    appear to the right of the previous word.

Positioning of target words fall under three cases:
(1) Words generated by NULL are distributed uniformly
(2) For a head word t, its position is modeled by the probability
    v_head(dv | max_v,word_class_t(t))
(3) For a non-head word t, its position is modeled by the probability
    v_non_head(dv | max_v,word_class_t(t))
dv and max_v are defined differently for head and non-head words.

The EM algorithm used in Model 5 is:
E step - In the training data, collect counts, weighted by prior
         probabilities.
         (a) count how many times a source language word is translated
             into a target language word
         (b) for a particular word class and maximum vacancy, count how
             many times a head word and the previous cept's center have
             a particular difference in number of vacancies
         (b) for a particular word class and maximum vacancy, count how
             many times a non-head word and the previous target word
             have a particular difference in number of vacancies
         (d) count how many times a source word is aligned to phi number
             of target words
         (e) count how many times NULL is aligned to a target word

M step - Estimate new probabilities based on the counts from the E step

Like Model 4, there are too many possible alignments to consider. Thus,
a hill climbing approach is used to sample good candidates. In addition,
pruning is used to weed out unlikely alignments based on Model 4 scores.


Notations:
i: Position in the source sentence
    Valid values are 0 (for NULL), 1, 2, ..., length of source sentence
j: Position in the target sentence
    Valid values are 1, 2, ..., length of target sentence
l: Number of words in the source sentence, excluding NULL
m: Number of words in the target sentence
s: A word in the source language
t: A word in the target language
phi: Fertility, the number of target words produced by a source word
p1: Probability that a target word produced by a source word is
    accompanied by another target word that is aligned to NULL
p0: 1 - p1
max_v: Maximum vacancy
dv: Vacancy difference, Δv

The definition of v_head here differs from GIZA++, section 4.7 of
[Brown et al., 1993], and [Koehn, 2010]. In the latter cases, v_head is
v_head(v(j) | v(center of previous cept),max_v,word_class(t)).

Here, we follow appendix B of [Brown et al., 1993] and combine v(j) with
v(center of previous cept) to obtain dv:
v_head(v(j) - v(center of previous cept) | max_v,word_class(t)).


References:
Philipp Koehn. 2010. Statistical Machine Translation.
Cambridge University Press, New York.

Peter E Brown, Stephen A. Della Pietra, Vincent J. Della Pietra, and
Robert L. Mercer. 1993. The Mathematics of Statistical Machine
Translation: Parameter Estimation. Computational Linguistics, 19 (2),
263-311.
"""

from __future__ import division
from collections import defaultdict
from math import factorial
from nltk.align import AlignedSent
from nltk.align import Alignment
from nltk.align import IBMModel
from nltk.align import IBMModel4
from nltk.align.ibm_model import Counts
from nltk.align.ibm_model import longest_target_sentence_length
import warnings


class IBMModel5(IBMModel):
    """
    Translation model that keeps track of vacant positions in the target
    sentence to decide where to place translated words

    >>> bitext = []
    >>> bitext.append(AlignedSent(['klein', 'ist', 'das', 'haus'], ['the', 'house', 'is', 'small']))
    >>> bitext.append(AlignedSent(['das', 'haus', 'ist', 'ja', 'groß'], ['the', 'house', 'is', 'big']))
    >>> bitext.append(AlignedSent(['das', 'buch', 'ist', 'ja', 'klein'], ['the', 'book', 'is', 'small']))
    >>> bitext.append(AlignedSent(['ein', 'haus', 'ist', 'klein'], ['a', 'house', 'is', 'small']))
    >>> bitext.append(AlignedSent(['das', 'haus'], ['the', 'house']))
    >>> bitext.append(AlignedSent(['das', 'buch'], ['the', 'book']))
    >>> bitext.append(AlignedSent(['ein', 'buch'], ['a', 'book']))
    >>> bitext.append(AlignedSent(['ich', 'fasse', 'das', 'buch', 'zusammen'], ['i', 'summarize', 'the', 'book']))
    >>> bitext.append(AlignedSent(['fasse', 'zusammen'], ['summarize']))
    >>> src_classes = {'the': 0, 'a': 0, 'small': 1, 'big': 1, 'house': 2, 'book': 2, 'is': 3, 'i': 4, 'summarize': 5 }
    >>> trg_classes = {'das': 0, 'ein': 0, 'haus': 1, 'buch': 1, 'klein': 2, 'groß': 2, 'ist': 3, 'ja': 4, 'ich': 5, 'fasse': 6, 'zusammen': 6 }

    >>> ibm5 = IBMModel5(bitext, 5, src_classes, trg_classes)

    >>> print('{0:.3f}'.format(ibm5.head_vacancy_table[1][1][1]))
    1.000
    >>> print('{0:.3f}'.format(ibm5.head_vacancy_table[2][1][1]))
    0.000
    >>> print('{0:.3f}'.format(ibm5.non_head_vacancy_table[3][3][6]))
    1.000

    >>> print('{0:.3f}'.format(ibm5.fertility_table[2]['summarize']))
    1.000
    >>> print('{0:.3f}'.format(ibm5.fertility_table[1]['book']))
    1.000

    >>> print('{0:.3f}'.format(ibm5.p1))
    0.033

    >>> test_sentence = bitext[2]
    >>> test_sentence.words
    ['das', 'buch', 'ist', 'ja', 'klein']
    >>> test_sentence.mots
    ['the', 'book', 'is', 'small']
    >>> test_sentence.alignment
    Alignment([(0, 0), (1, 1), (2, 2), (3, None), (4, 3)])

    """
    MIN_SCORE_FACTOR = 0.2
    """
    Alignments with scores below this factor are pruned during sampling
    """

    def __init__(self, sentence_aligned_corpus, iterations,
                 source_word_classes, target_word_classes,
                 probability_tables=None):
        """
        Train on ``sentence_aligned_corpus`` and create a lexical
        translation model, vacancy models, a fertility model, and a
        model for generating NULL-aligned words.

        Translation direction is from ``AlignedSent.mots`` to
        ``AlignedSent.words``.

        :param sentence_aligned_corpus: Sentence-aligned parallel corpus
        :type sentence_aligned_corpus: list(AlignedSent)

        :param iterations: Number of iterations to run training algorithm
        :type iterations: int

        :param source_word_classes: Lookup table that maps a source word
            to its word class, the latter represented by an integer id
        :type source_word_classes: dict[str]: int

        :param target_word_classes: Lookup table that maps a target word
            to its word class, the latter represented by an integer id
        :type target_word_classes: dict[str]: int

        :param probability_tables: Optional. Use this to pass in custom
            probability values. If not specified, probabilities will be
            set to a uniform distribution, or some other sensible value.
            If specified, all the following entries must be present:
            ``translation_table``, ``alignment_table``,
            ``fertility_table``, ``p1``, ``head_distortion_table``,
            ``non_head_distortion_table``, ``head_vacancy_table``,
            ``non_head_vacancy_table``. See ``IBMModel``, ``IBMModel4``,
            and ``IBMModel5`` for the type and purpose of these tables.
        :type probability_tables: dict[str]: object
        """
        super(IBMModel5, self).__init__(sentence_aligned_corpus)
        self.reset_probabilities()
        self.src_classes = source_word_classes
        self.trg_classes = target_word_classes

        if probability_tables is None:
            # Get probabilities from IBM model 4
            ibm4 = IBMModel4(sentence_aligned_corpus, iterations,
                             source_word_classes, target_word_classes)
            self.translation_table = ibm4.translation_table
            self.alignment_table = ibm4.alignment_table
            self.fertility_table = ibm4.fertility_table
            self.p1 = ibm4.p1
            self.head_distortion_table = ibm4.head_distortion_table
            self.non_head_distortion_table = ibm4.non_head_distortion_table
            self.set_uniform_distortion_probabilities(sentence_aligned_corpus)
        else:
            # Set user-defined probabilities
            self.translation_table = probability_tables['translation_table']
            self.alignment_table = probability_tables['alignment_table']
            self.fertility_table = probability_tables['fertility_table']
            self.p1 = probability_tables['p1']
            self.head_distortion_table = probability_tables[
                'head_distortion_table']
            self.non_head_distortion_table = probability_tables[
                'non_head_distortion_table']
            self.head_vacancy_table = probability_tables[
                'head_vacancy_table']
            self.non_head_vacancy_table = probability_tables[
                'non_head_vacancy_table']

        for k in range(0, iterations):
            self.train(sentence_aligned_corpus)

    def reset_probabilities(self):
        super(IBMModel5, self).reset_probabilities()
        self.head_vacancy_table = defaultdict(
            lambda: defaultdict(lambda: defaultdict(lambda: self.MIN_PROB)))
        """
        dict[int][int][int]: float. Probability(vacancy difference |
        number of remaining valid positions,target word class).
        Values accessed as ``head_vacancy_table[dv][v_max][trg_class]``.
        """

        self.non_head_vacancy_table = defaultdict(
            lambda: defaultdict(lambda: defaultdict(lambda: self.MIN_PROB)))
        """
        dict[int][int][int]: float. Probability(vacancy difference |
        number of remaining valid positions,target word class).
        Values accessed as ``non_head_vacancy_table[dv][v_max][trg_class]``.
        """

    def set_uniform_distortion_probabilities(self, sentence_aligned_corpus):
        """
        Set vacancy probabilities uniformly to
        1 / cardinality of vacancy difference values
        """
        max_m = longest_target_sentence_length(sentence_aligned_corpus)

        # The maximum vacancy difference occurs when a word is placed in
        # the last available position m of the target sentence and the
        # previous word position has no vacancies.
        # The minimum is 1-max_v, when a word is placed in the first
        # available position and the previous word is placed beyond the
        # last available position.
        # Thus, the number of possible vacancy difference values is
        # (max_v) - (1-max_v) + 1 = 2 * max_v.
        if max_m > 0 and (float(1) / (2 * max_m)) < IBMModel.MIN_PROB:
            warnings.warn("A target sentence is too long (" + str(max_m) +
                          " words). Results may be less accurate.")

        for max_v in range(1, max_m + 1):
            for dv in range(1, max_m + 1):
                initial_prob = 1 / (2 * max_v)
                self.head_vacancy_table[dv][max_v] = defaultdict(
                    lambda: initial_prob)
                self.head_vacancy_table[-(dv-1)][max_v] = defaultdict(
                    lambda: initial_prob)
                self.non_head_vacancy_table[dv][max_v] = defaultdict(
                    lambda: initial_prob)
                self.non_head_vacancy_table[-(dv-1)][max_v] = defaultdict(
                    lambda: initial_prob)

    def train(self, parallel_corpus):
        # Reset all counts
        counts = Model5Counts()

        for aligned_sentence in parallel_corpus:
            l = len(aligned_sentence.mots)
            m = len(aligned_sentence.words)

            # Sample the alignment space
            sampled_alignments, best_alignment = self.sample(aligned_sentence)
            # Record the most probable alignment
            aligned_sentence.alignment = Alignment(
                best_alignment.zero_indexed_alignment())

            # E step (a): Compute normalization factors to weigh counts
            total_count = self.prob_of_alignments(sampled_alignments)

            # E step (b): Collect counts
            for alignment_info in sampled_alignments:
                count = self.prob_t_a_given_s(alignment_info)
                normalized_count = count / total_count

                for j in range(1, m + 1):
                    counts.update_lexical_translation(
                        normalized_count, alignment_info, j)

                slots = Slots(m)
                for i in range(1, l + 1):
                    counts.update_vacancy(
                        normalized_count, alignment_info, i,
                        self.trg_classes, slots)

                counts.update_null_generation(normalized_count, alignment_info)
                counts.update_fertility(normalized_count, alignment_info)

        # M step: Update probabilities with maximum likelihood estimates
        # If any probability is less than MIN_PROB, clamp it to MIN_PROB
        existing_alignment_table = self.alignment_table
        self.reset_probabilities()
        # don't retrain alignment table
        self.alignment_table = existing_alignment_table

        self.maximize_lexical_translation_probabilities(counts)
        self.maximize_vacancy_probabilities(counts)
        self.maximize_fertility_probabilities(counts)
        self.maximize_null_generation_probabilities(counts)

    def sample(self, sentence_pair):
        """
        Sample the most probable alignments from the entire alignment
        space according to Model 4

        Note that Model 4 scoring is used instead of Model 5 because the
        latter is too expensive to compute.

        First, determine the best alignment according to IBM Model 2.
        With this initial alignment, use hill climbing to determine the
        best alignment according to a IBM Model 4. Add this
        alignment and its neighbors to the sample set. Repeat this
        process with other initial alignments obtained by pegging an
        alignment point. Finally, prune alignments that have
        substantially lower Model 4 scores than the best alignment.

        :param sentence_pair: Source and target language sentence pair
            to generate a sample of alignments from
        :type sentence_pair: AlignedSent

        :return: A set of best alignments represented by their ``AlignmentInfo``
            and the best alignment of the set for convenience
        :rtype: set(AlignmentInfo), AlignmentInfo
        """
        sampled_alignments, best_alignment = super(
            IBMModel5, self).sample(sentence_pair)
        return self.prune(sampled_alignments), best_alignment

    def prune(self, alignment_infos):
        """
        Removes alignments from ``alignment_infos`` that have
        substantially lower Model 4 scores than the best alignment

        :return: Pruned alignments
        :rtype: set(AlignmentInfo)
        """
        alignments = []
        best_score = 0

        for alignment_info in alignment_infos:
            score = IBMModel4.model4_prob_t_a_given_s(alignment_info, self)
            best_score = max(score, best_score)
            alignments.append((alignment_info, score))

        threshold = IBMModel5.MIN_SCORE_FACTOR * best_score
        alignments = [a[0] for a in alignments if a[1] > threshold]
        return set(alignments)

    def hillclimb(self, alignment_info, j_pegged=None):
        """
        Starting from the alignment in ``alignment_info``, look at
        neighboring alignments iteratively for the best one, according
        to Model 4

        Note that Model 4 scoring is used instead of Model 5 because the
        latter is too expensive to compute.

        There is no guarantee that the best alignment in the alignment
        space will be found, because the algorithm might be stuck in a
        local maximum.

        :param j_pegged: If specified, the search will be constrained to
            alignments where ``j_pegged`` remains unchanged
        :type j_pegged: int

        :return: The best alignment found from hill climbing
        :rtype: AlignmentInfo
        """
        alignment = alignment_info  # alias with shorter name
        max_probability = IBMModel4.model4_prob_t_a_given_s(alignment, self)

        while True:
            old_alignment = alignment
            for neighbor_alignment in self.neighboring(alignment, j_pegged):
                neighbor_probability = IBMModel4.model4_prob_t_a_given_s(
                    neighbor_alignment, self)

                if neighbor_probability > max_probability:
                    alignment = neighbor_alignment
                    max_probability = neighbor_probability

            if alignment == old_alignment:
                # Until there are no better alignments
                break

        alignment.score = max_probability
        return alignment

    def prob_t_a_given_s(self, alignment_info):
        """
        Probability of target sentence and an alignment given the
        source sentence
        """
        probability = 1.0
        MIN_PROB = IBMModel.MIN_PROB
        slots = Slots(len(alignment_info.trg_sentence) - 1)

        def null_generation_term():
            # Binomial distribution: B(m - null_fertility, p1)
            value = 1.0
            p1 = self.p1
            p0 = 1 - p1
            null_fertility = alignment_info.fertility_of_i(0)
            m = len(alignment_info.trg_sentence) - 1
            value *= (pow(p1, null_fertility) * pow(p0, m - 2 * null_fertility))
            if value < MIN_PROB:
                return MIN_PROB

            # Combination: (m - null_fertility) choose null_fertility
            for i in range(1, null_fertility + 1):
                value *= (m - null_fertility - i + 1) / i
            return value

        def fertility_term():
            value = 1.0
            src_sentence = alignment_info.src_sentence
            for i in range(1, len(src_sentence)):
                fertility = alignment_info.fertility_of_i(i)
                value *= (factorial(fertility) *
                          self.fertility_table[fertility][src_sentence[i]])
                if value < MIN_PROB:
                    return MIN_PROB
            return value

        def lexical_translation_term(j):
            t = alignment_info.trg_sentence[j]
            i = alignment_info.alignment[j]
            s = alignment_info.src_sentence[i]
            return self.translation_table[t][s]

        def vacancy_term(i):
            value = 1.0
            tablet = alignment_info.cepts[i]
            tablet_length = len(tablet)
            total_vacancies = slots.vacancies_at(len(slots))

            # case 1: NULL-aligned words
            if tablet_length == 0:
                return value

            # case 2: head word
            j = tablet[0]
            previous_cept = alignment_info.previous_cept(j)
            previous_center = alignment_info.center_of_cept(previous_cept)
            dv = slots.vacancies_at(j) - slots.vacancies_at(previous_center)
            max_v = total_vacancies - tablet_length + 1
            trg_class = self.trg_classes[alignment_info.trg_sentence[j]]
            value *= self.head_vacancy_table[dv][max_v][trg_class]
            slots.occupy(j)  # mark position as occupied
            total_vacancies -= 1
            if value < MIN_PROB:
                return MIN_PROB

            # case 3: non-head words
            for k in range(1, tablet_length):
                previous_position = tablet[k - 1]
                previous_vacancies = slots.vacancies_at(previous_position)
                j = tablet[k]
                dv = slots.vacancies_at(j) - previous_vacancies
                max_v = (total_vacancies - tablet_length + k + 1 -
                         previous_vacancies)
                trg_class = self.trg_classes[alignment_info.trg_sentence[j]]
                value *= self.non_head_vacancy_table[dv][max_v][trg_class]
                slots.occupy(j)  # mark position as occupied
                total_vacancies -= 1
                if value < MIN_PROB:
                    return MIN_PROB

            return value
        # end nested functions

        # Abort computation whenever probability falls below MIN_PROB at
        # any point, since MIN_PROB can be considered as zero
        probability *= null_generation_term()
        if probability < MIN_PROB:
            return MIN_PROB

        probability *= fertility_term()
        if probability < MIN_PROB:
            return MIN_PROB

        for j in range(1, len(alignment_info.trg_sentence)):
            probability *= lexical_translation_term(j)
            if probability < MIN_PROB:
                return MIN_PROB

        for i in range(1, len(alignment_info.src_sentence)):
            probability *= vacancy_term(i)
            if probability < MIN_PROB:
                return MIN_PROB

        return probability

    def maximize_vacancy_probabilities(self, counts):
        MIN_PROB = IBMModel.MIN_PROB
        head_vacancy_table = self.head_vacancy_table
        for dv, max_vs in counts.head_vacancy.items():
            for max_v, trg_classes in max_vs.items():
                for t_cls in trg_classes:
                    estimate = (counts.head_vacancy[dv][max_v][t_cls] /
                                counts.head_vacancy_for_any_dv[max_v][t_cls])
                    head_vacancy_table[dv][max_v][t_cls] = max(estimate,
                                                               MIN_PROB)

        non_head_vacancy_table = self.non_head_vacancy_table
        for dv, max_vs in counts.non_head_vacancy.items():
            for max_v, trg_classes in max_vs.items():
                for t_cls in trg_classes:
                    estimate = (
                        counts.non_head_vacancy[dv][max_v][t_cls] /
                        counts.non_head_vacancy_for_any_dv[max_v][t_cls])
                    non_head_vacancy_table[dv][max_v][t_cls] = max(estimate,
                                                                   MIN_PROB)


class Model5Counts(Counts):
    """
    Data object to store counts of various parameters during training.
    Include counts for vacancies.
    """
    def __init__(self):
        super(Model5Counts, self).__init__()
        self.head_vacancy = defaultdict(
            lambda: defaultdict(lambda: defaultdict(lambda: 0.0)))
        self.head_vacancy_for_any_dv = defaultdict(
            lambda: defaultdict(lambda: 0.0))
        self.non_head_vacancy = defaultdict(
            lambda: defaultdict(lambda: defaultdict(lambda: 0.0)))
        self.non_head_vacancy_for_any_dv = defaultdict(
            lambda: defaultdict(lambda: 0.0))

    def update_vacancy(self, count, alignment_info, i, trg_classes, slots):
        """
        :param count: Value to add to the vacancy counts
        :param alignment_info: Alignment under consideration
        :param i: Source word position under consideration
        :param trg_classes: Target word classes
        :param slots: Vacancy states of the slots in the target sentence.
            Output parameter that will be modified as new words are placed
            in the target sentence.
        """
        tablet = alignment_info.cepts[i]
        tablet_length = len(tablet)
        total_vacancies = slots.vacancies_at(len(slots))

        # case 1: NULL aligned words
        if tablet_length == 0:
            return  # ignore zero fertility words

        # case 2: head word
        j = tablet[0]
        previous_cept = alignment_info.previous_cept(j)
        previous_center = alignment_info.center_of_cept(previous_cept)
        dv = slots.vacancies_at(j) - slots.vacancies_at(previous_center)
        max_v = total_vacancies - tablet_length + 1
        trg_class = trg_classes[alignment_info.trg_sentence[j]]
        self.head_vacancy[dv][max_v][trg_class] += count
        self.head_vacancy_for_any_dv[max_v][trg_class] += count
        slots.occupy(j)  # mark position as occupied
        total_vacancies -= 1

        # case 3: non-head words
        for k in range(1, tablet_length):
            previous_position = tablet[k - 1]
            previous_vacancies = slots.vacancies_at(previous_position)
            j = tablet[k]
            dv = slots.vacancies_at(j) - previous_vacancies
            max_v = (total_vacancies - tablet_length + k + 1 -
                     previous_vacancies)
            trg_class = trg_classes[alignment_info.trg_sentence[j]]
            self.non_head_vacancy[dv][max_v][trg_class] += count
            self.non_head_vacancy_for_any_dv[max_v][trg_class] += count
            slots.occupy(j)  # mark position as occupied
            total_vacancies -= 1


class Slots(object):
    """
    Represents positions in a target sentence. Used to keep track of
    which slot (position) is occupied.
    """
    def __init__(self, target_sentence_length):
        self._slots = [False] * (target_sentence_length + 1)  # 1-indexed

    def occupy(self, position):
        """
        :return: Mark slot at ``position`` as occupied
        """
        self._slots[position] = True

    def vacancies_at(self, position):
        """
        :return: Number of vacant slots up to, and including, ``position``
        """
        vacancies = 0
        for k in range(1, position + 1):
            if not self._slots[k]:
                vacancies += 1
        return vacancies

    def __len__(self):
        return len(self._slots) - 1  # exclude dummy zeroeth element
