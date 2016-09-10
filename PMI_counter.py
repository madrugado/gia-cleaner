# coding: utf-8
import os
from os import path as osp

from nltk.tokenize import word_tokenize
from nltk.util import ngrams, bigrams, trigrams, skipgrams
from nltk.corpus import stopwords
from collections import defaultdict
from itertools import product, combinations
from math import log
from time import clock, time
from multiprocessing import Pool
import operator
import re
from nltk.stem.snowball import SnowballStemmer
from gensim.models import Word2Vec
import sys
import codecs

pat = re.compile('[^\w ]')
stemmer = SnowballStemmer('russian')

NUM_WORKERS = 10

DATA_PATH = 'data'
DOCS_DIR = osp.join(DATA_PATH, 'wikipedia_content_based_on_GIA_keyword_one_file_per_keyword_search_clean')
TRAIN_SET_PATH = osp.join(DATA_PATH, 'gia-corpus/tsv_tests/all_questions.tsv')
VALIDATION_SET_PATH = osp.join(DATA_PATH, 'validation_set.tsv')


class QAJointNgrams:
    def __init__(self, qa_joint_ngrams_A, qa_joint_ngrams_B, qa_joint_ngrams_C, qa_joint_ngrams_D):
        self.A = qa_joint_ngrams_A
        self.B = qa_joint_ngrams_B
        self.C = qa_joint_ngrams_C
        self.D = qa_joint_ngrams_D
        self.all_ngrams = set(self.A) | set(self.B) | set(self.C) | set(self.D)

class Answers:
    def __init__(self, answerA, answerB, answerC, answerD):
        self.A = answerA
        self.B = answerB
        self.C = answerC
        self.D = answerD

class NGramsContainer:
    global stopwords
    skip_for_skipgram = 8

    def __init__(self, text, tokenize=True):
        if tokenize:
            text = word_tokenize(text)

        text = map(stemmer.stem, text)

        self.untouched_text = list(text)

        self.unigrams = self._get_filtered_unigrams(text)
        self.bigrams = self._get_filtered_bigrams(text)
        self.trigrams = self._get_filtered_trigrams(text)
        self.skipbigrams = self._get_filtered_skipbigrams(text)

        self.all_ngrams = set(self.unigrams) | set(self.bigrams) | set(self.trigrams) | set(self.skipbigrams)

    def _get_filtered_unigrams(self, words):
        filtered_unigrams = []
        for w in words:
            if w not in stopwords:
                filtered_unigrams.append(w)
        return filtered_unigrams

    def _get_filtered_bigrams(self, words):
        filtered_bigrams = []

        for bi in bigrams(words):
            if not any(w for w in bi if w in stopwords) and bi[0] != bi[1]:
                filtered_bigrams.append(bi)
        return filtered_bigrams

    def _get_filtered_trigrams(self, words):
        # Allow stopword in the middle of trigram
        filtered_trigrams = []
        for tri in trigrams(words):
            leave = True
            for i, w in enumerate(tri):
                if w in stopwords and i != 1:
                    leave = False
                    break
            if leave and tri[0] != tri[1] and tri[1] != tri[2]:
                filtered_trigrams.append(tri)
        return filtered_trigrams

    def _get_filtered_skipbigrams(self, words):
        filtered_skipped_bigrams = []
        for bi in skipgrams(words, 2, NGramsContainer.skip_for_skipgram):
            if not any(w for w in bi if w in stopwords) and bi[0] != bi[1]:
                filtered_skipped_bigrams.append(bi)
        return filtered_skipped_bigrams

    def add_next_word(self, word):
        #word = lemmatizer.lemmatize(word)
        word = stemmer.stem(word)
        old_unigrams = self._get_filtered_unigrams((self.untouched_text[0],))
        if old_unigrams:
            assert(len(old_unigrams) == 1)
            self.unigrams.remove(old_unigrams[0])

        old_bigrams = self._get_filtered_bigrams(self.untouched_text[:2])
        if old_bigrams:
            assert(len(old_bigrams) == 1)
            self.bigrams.remove(old_bigrams[0])

        old_trigrams = self._get_filtered_trigrams(self.untouched_text[:3])
        if old_trigrams:
            assert(len(old_trigrams) == 1)
            self.trigrams.remove(old_trigrams[0])

        old_skipbigrams = list(product([self.untouched_text[0]], self.untouched_text[1:NGramsContainer.skip_for_skipgram + 2]))
        filtered_old_skipped_bigrams = []
        for bi in old_skipbigrams:
            if not any(w for w in bi if w in stopwords) and bi[0] != bi[1]:
                filtered_old_skipped_bigrams.append(bi)
        for skipgram in filtered_old_skipped_bigrams:
            self.skipbigrams.remove(skipgram)

        new_unigram = self._get_filtered_unigrams((word,))
        if new_unigram:
            self.unigrams += new_unigram

        new_bigram = self._get_filtered_bigrams([self.untouched_text[-1], word])
        if new_bigram:
            self.bigrams += new_bigram

        new_trigram = self._get_filtered_trigrams([self.untouched_text[-2], self.untouched_text[-1], word])
        if new_trigram:
            self.trigrams += new_trigram

        new_skipbigrams = list(product(self.untouched_text[-1 * (NGramsContainer.skip_for_skipgram + 1):], [word]))
        for bi in new_skipbigrams:
            if not any(w for w in bi if w in stopwords) and bi[0] != bi[1]:
                self.skipbigrams.append(bi)

        removed_word = self.untouched_text.pop(0)
        self.untouched_text.append(word)
        self.all_ngrams = set(self.unigrams) | set(self.bigrams) | set(self.trigrams) | set(self.skipbigrams)


def get_ngram_set_from_attr(attr_name, items):
    result_ngrams = set()
    for item in items:
        attr = getattr(item, attr_name)
        result_ngrams.update(attr)
    return result_ngrams


def count_ngrams(ngrams_to_count, source, count_dict):
    for ngram in ngrams_to_count:
        if ngram in source:
            count_dict[ngram] += 1


def count_joint_ngrams(qa_joint_ngrams, all_window_ngrams, counter):
    for q_ngram, a_ngram in qa_joint_ngrams:
        if q_ngram in all_window_ngrams and a_ngram in all_window_ngrams:
            counter[(q_ngram, a_ngram)] += 1


def compute_PMI(joint_qa_ndgrams, global_ngram_count, global_joint_ngram_count, number_of_windows):
    PMI = defaultdict(float)

    for joint_ngram in joint_qa_ndgrams:
        joint_freq = global_joint_ngram_count[joint_ngram] / float(number_of_windows)
        q_ngram, a_ngram = joint_ngram
        q_ngram_freq = global_ngram_count[q_ngram] / float(number_of_windows)
        a_ngram_freq = global_ngram_count[a_ngram] / float(number_of_windows)
        if q_ngram_freq > 0 and a_ngram_freq > 0 and joint_freq > 0:
            PMI[joint_ngram] = max(0, log(joint_freq / (q_ngram_freq * a_ngram_freq)))
            print collection_str(joint_ngram), PMI[joint_ngram]
        else:
            PMI[joint_ngram] = 0
    return PMI


def get_average_PMI(PMI_dict):
    if len(PMI_dict) > 0:
        return sum(PMI_dict[key] for key in PMI_dict) / len(PMI_dict)
    else:
        return 0.0


def test():
    with open('../data/new_lucene_index/new_lucene_index/ck12concepts/2535_33.html.txt', 'r') as doc:
        text = doc.readlines()
        text = ' '.join([x.strip().decode('unicode_escape').encode('ascii', 'ignore') for x in text])  # concatenate whole file into one string
        text = ''.join([x for x in text if x not in ':.,?()+"'])

        text_words = word_tokenize(text.lower())

        window_size = 10

        for i, window in enumerate(list(ngrams(text_words, window_size))):
            if i == 0:
                window_ngrams = NGramsContainer(window, tokenize=False)
            else:

                true_window_ngrams = NGramsContainer(window, tokenize=False)
                window_ngrams.add_next_word(window[-1])

                #print sorted(window_ngrams.skipbigrams)
                #print sorted(true_window_ngrams.skipbigrams)

                # assert(window_ngrams.all_ngrams == true_window_ngrams.all_ngrams)
                assert window_ngrams.unigrams == true_window_ngrams.unigrams, i
                assert window_ngrams.bigrams == true_window_ngrams.bigrams, i
                assert window_ngrams.trigrams == true_window_ngrams.trigrams, i
                assert sorted(window_ngrams.skipbigrams) == sorted(true_window_ngrams.skipbigrams), i
                # Order of skipbigrams is not the same, but who cares


def get_extended_question(question):
    return question
    sentence = pat.sub('', question)
    result = map(stemmer.stem, sentence.lower().split())
    new_result = []
    for word in result:
        new_result.append(word)
        for additional in model.most_similar(word)[:2]:
            new_result.append(additional)
    return ' '.join(new_result)


def extract_ngrams_from_question_and_answers(question, answers):
    question_ngrams = NGramsContainer(question.lower())
    answerA_ngrams = NGramsContainer(
        get_extended_question(answers.A.lower()))
    answerB_ngrams = NGramsContainer(
        get_extended_question(answers.B.lower()))
    answerC_ngrams = NGramsContainer(
        get_extended_question(answers.C.lower()))
    answerD_ngrams = NGramsContainer(
        get_extended_question(answers.D.lower()))

    qa_ngrams = question_ngrams.all_ngrams | answerA_ngrams.all_ngrams | answerB_ngrams.all_ngrams \
        | answerC_ngrams.all_ngrams | answerD_ngrams.all_ngrams

    all_q_ngrams = question_ngrams.all_ngrams

    qa_joint_ngrams = QAJointNgrams(
            list(product(all_q_ngrams, answerA_ngrams.all_ngrams)),
            list(product(all_q_ngrams, answerB_ngrams.all_ngrams)),
            list(product(all_q_ngrams, answerC_ngrams.all_ngrams)),
            list(product(all_q_ngrams, answerD_ngrams.all_ngrams)))

    return qa_ngrams, qa_joint_ngrams


def predict_answer(joint_ngrams_for_each_answer, global_ngram_count, global_joint_ngram_count, number_of_windows):
    print 'PMI A'
    PMI_A = compute_PMI(joint_ngrams_for_each_answer.A, global_ngram_count, global_joint_ngram_count, number_of_windows)
    print 'PMI B'
    PMI_B = compute_PMI(joint_ngrams_for_each_answer.B, global_ngram_count, global_joint_ngram_count, number_of_windows)
    print 'PMI C'
    PMI_C = compute_PMI(joint_ngrams_for_each_answer.C, global_ngram_count, global_joint_ngram_count, number_of_windows)
    print 'PMI D'
    PMI_D = compute_PMI(joint_ngrams_for_each_answer.D, global_ngram_count, global_joint_ngram_count, number_of_windows)

    average_PMI_A = get_average_PMI(PMI_A)
    average_PMI_B = get_average_PMI(PMI_B)
    average_PMI_C = get_average_PMI(PMI_C)
    average_PMI_D = get_average_PMI(PMI_D)

    max_PMI = max(average_PMI_A, average_PMI_B, average_PMI_C, average_PMI_D)
    if max_PMI == average_PMI_A:
        ans = 'A'
    elif max_PMI == average_PMI_B:
        ans = 'B'
    elif max_PMI == average_PMI_C:
        ans = 'C'
    elif max_PMI == average_PMI_D:
        ans = 'D'

    return ans, average_PMI_A, average_PMI_B, average_PMI_C, average_PMI_D


def extract_ngram_counts_from_text(f_path):
    ngram_count = defaultdict(int)
    joint_ngram_count = defaultdict(int)
    number_of_windows = 0

    with codecs.open(f_path, 'r', 'utf-8') as doc:
        text = doc.readlines()
        text = ' '.join([x.strip() for x in text])  # concatenate whole file into one string
        text = ''.join([x for x in text if x not in u'—;:.,?()+"_-%'])

        text_words = word_tokenize(text.lower())

        first = True

        for i, window in enumerate(ngrams(text_words, window_size)):
            if first:
                window_ngrams = NGramsContainer(window, tokenize=False)
                first = False
            else:
                window_ngrams.add_next_word(window[-1])
                #true_window_ngrams = NGramsContainer(window, tokenize=False)

                #assert true_window_ngrams.all_ngrams == window_ngrams.all_ngrams

            number_of_windows += 1
            all_window_ngrams = window_ngrams.all_ngrams
            all_windows_ngrams_combinations = list(combinations(all_window_ngrams, 2))

            # Compute independent ngram counts
            for ngram in all_window_ngrams:
                if ngram in global_ngrams:
                    ngram_count[ngram] += 1
            for qa_ngram in all_windows_ngrams_combinations:
                if qa_ngram in global_joint_ngrams:
                    joint_ngram_count[qa_ngram] += 1

        return (ngram_count, joint_ngram_count, number_of_windows)

def collection_str(collection):
    if isinstance(collection, list):
        brackets = u'[%s]'
        single_add = u''
    elif isinstance(collection, tuple):
        brackets = u'(%s)'
        single_add = u','
    else:
        return unicode(collection)
    items = u', '.join([collection_str(x) for x in collection])
    if len(collection) == 1:
        items += single_add
    return brackets % items

if __name__ == '__main__':
    stopwords = set(map(stemmer.stem, stopwords.words('russian')))

    #test()
    print 'test ok'

    true_answers = dict()
    predicted_answers = dict()
    questions = dict()
    answers = dict()
    train_question_ids = []
    val_question_ids = []

    global_joint_ngrams_for_each_answer = dict()

    global_ngram_count = defaultdict(int)
    global_joint_ngram_count = defaultdict(int)
    global_ngrams = set()
    global_joint_ngrams = set()

    with codecs.open(TRAIN_SET_PATH, 'r', 'utf-8') as train_file:
        train_questions = [x.strip() for x in train_file.readlines()]

        qa_ngram_prep_time = clock()
        for s in train_questions[1:]:
            id, question, right_answer, answerA, answerB, answerC, answerD = s.split('\t')
            train_question_ids.append(id)
            true_answers[id] = right_answer
            questions[id] = question
            answers[id] = Answers(answerA, answerB, answerC, answerD)

            qa_ngrams, qa_joint_ngrams = extract_ngrams_from_question_and_answers(questions[id], answers[id])
            global_ngrams.update(qa_ngrams)
            global_joint_ngrams_for_each_answer[id] = qa_joint_ngrams
            global_joint_ngrams.update(global_joint_ngrams_for_each_answer[id].all_ngrams)

    # TODO: remove code duplication with prev parsing.
    """
    with open(VALIDATION_SET_PATH, 'r') as train_file:
        val_questions = [x.strip() for x in train_file.readlines()]

        qa_ngram_prep_time = clock()
        for s in val_questions[1:]:
            id, question, answerA, answerB, answerC, answerD = s.split('\t')
            val_question_ids.append(id)

            assert id not in questions
            assert id not in answers

            questions[id] = question
            answers[id] = Answers(answerA, answerB, answerC, answerD)

            qa_ngrams, qa_joint_ngrams = extract_ngrams_from_question_and_answers(questions[id], answers[id])
            global_ngrams.update(qa_ngrams)
            global_joint_ngrams_for_each_answer[id] = qa_joint_ngrams
            global_joint_ngrams.update(global_joint_ngrams_for_each_answer[id].all_ngrams)
    """

    print 'prepared', len(global_joint_ngrams), 'ngrams for', clock() - qa_ngram_prep_time

    global_number_of_windows = 0
    window_size = 10

    ngram_freq_extraction_time = time()

    files_to_process = []
    for root, dirs, files in os.walk(DOCS_DIR):
        for f in files:
            f_path = osp.join(root, f)
            files_to_process.append(f_path)

    files_to_process = files_to_process[:10]
    print len(files_to_process), 'files to process'

    ngram_extractors = Pool(NUM_WORKERS)
    ngram_counts_per_file = ngram_extractors.map(extract_ngram_counts_from_text, files_to_process)

    # Gather ngram counts from all files.
    for ngram_counts_for_one_file in ngram_counts_per_file:
        ngram_count, joint_ngram_count, number_of_windows = ngram_counts_for_one_file
        global_number_of_windows += number_of_windows
        for ngram in ngram_count:
            global_ngram_count[ngram] += ngram_count[ngram]
        for joint_ngram in joint_ngram_count:
            global_joint_ngram_count[joint_ngram] += joint_ngram_count[joint_ngram]

    print 'processed', len(files_to_process), 'files for', time() - ngram_freq_extraction_time

    sorted_ngram_counts = sorted(global_ngram_count.items(), key=operator.itemgetter(1), reverse=True)
    sorted_joint_ngram_counts = sorted(global_joint_ngram_count.items(), key=operator.itemgetter(1), reverse=True)

    print 'top 30 ngrams'
    for i in range(30):
        print collection_str(sorted_ngram_counts[i])

    print '\ntop 30 joint ngrams'
    for i in range(30):
        print collection_str(sorted_joint_ngram_counts[i])

    """
    print 'VALIDATION'
    with open('result.csv', 'w') as result:
        result.write('id,correctAnswer\n')
        with open('data/val_PMI_scores.csv', 'w') as pmi_scores:
            pmi_scores.write('id,PMI_score_A,PMI_score_B,PMI_score_C,PMI_score_D\n')

            for id in sorted(val_question_ids):
                print 'Q:', questions[id]
                print 'A:', answers[id].A
                print 'B:', answers[id].B
                print 'C:', answers[id].C
                print 'D:', answers[id].D

                ans, average_PMI_A, average_PMI_B, average_PMI_C, average_PMI_D = predict_answer(
                        global_joint_ngrams_for_each_answer[id], global_ngram_count, global_joint_ngram_count, global_number_of_windows)

                print 'average PMIs A:', average_PMI_A, 'B:', average_PMI_B, 'C:', average_PMI_C, 'D:', average_PMI_D
                print 'predicted ans:', ans

                pmi_scores.write('{0},{1},{2},{3},{4}\n'.format(id, average_PMI_A, average_PMI_B, average_PMI_C, average_PMI_D))
                result.write(id + ',' + ans + '\n')
    """

    print 'TRAIN'
    with codecs.open('data/train_PMI_scores.csv', 'w', 'utf-8') as pmi_scores:
        pmi_scores.write('id,PMI_score_A,PMI_score_B,PMI_score_C,PMI_score_D\n')

        for id in sorted(train_question_ids):
            print 'Q:', questions[id]
            print 'A:', answers[id].A
            print 'B:', answers[id].B
            print 'C:', answers[id].C
            print 'D:', answers[id].D

            ans, average_PMI_A, average_PMI_B, average_PMI_C, average_PMI_D = predict_answer(
                    global_joint_ngrams_for_each_answer[id], global_ngram_count, global_joint_ngram_count, global_number_of_windows)

            print 'average PMIs A:', average_PMI_A, 'B:', average_PMI_B, 'C:', average_PMI_C, 'D:', average_PMI_D
            print 'true answer:', true_answers[id]
            print 'predicted answer:', ans

            pmi_scores.write('{0},{1},{2},{3},{4}\n'.format(id, average_PMI_A, average_PMI_B, average_PMI_C, average_PMI_D))
            predicted_answers[id] = ans

    # print 'Question:', questions[id]
    # print 'A:', answersA[id], 'PMI', average_PMI_A
    # print 'B:', answersB[id], 'PMI', average_PMI_B
    # print 'C:', answersC[id], 'PMI', average_PMI_C
    # print 'D:', answersD[id], 'PMI', average_PMI_D
    # print 'ans:', ans

    count_of_right_answers = 0
    for id in true_answers.keys():
        if predicted_answers[id] == true_answers[id]:
            count_of_right_answers += 1
    acc = float(count_of_right_answers) / float(len(true_answers))
    print 'Accuracy on training set', acc
