from __future__ import print_function
import os
import time
import lucene
import numpy as np
import pandas as pd
import pickle

from lupyne import engine
from multiprocessing import Pool
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from itertools import chain
import codecs
import random
from timeit import default_timer

random.seed(0)

DATA_PATH = './data'
VAL_SET_PATH = './data/validation_set.tsv'
TRAIN_SET_PATH = '../data/gia-corpus/tsv_tests/all_questions.tsv'
CK_KEYWORDS_PATH = '../data/GIA_combined.txt'
DOCS_DIR = '../data/wikipedia_content_based_on_GIA_keyword_one_file_per_keyword_search_clean'


def prepare_index(docs_dir, stemmer, min_len=5):
    indexer = engine.Indexer()
    indexer.set('contents', stored=True)
    indexer.set('path', stored=True)

    p = Pool(4)
    index_start, indexed_documents_count = default_timer(), 0
    for root, dirs, files in os.walk(docs_dir):
        for f in files:
            if True or random.choice([0, 1, 2]):
                print(f)
                with codecs.open(os.path.join(root, f), 'r', 'utf-8') as doc:
                    articles = filter(lambda x: len(x.split()) >= min_len, doc.readlines())
                    stemmed_articles = p.map(stem_sentence_and_remove_stopwords, articles)
                    for article in stemmed_articles:
                        indexer.add(path=os.path.join(root, f), contents=article)
                        indexed_documents_count += 1

                    """ for article in doc.readlines():
                        if len(article.split()) >= min_len:
                            article = stem_sentence_and_remove_stopwords(article)
                            indexer.add(path=os.path.join(root, f), contents=article)
                            indexed_documents_count += 1
                    """
                #break
    indexer.commit()
    print('Added', indexed_documents_count, 'documents to index, spent', default_timer() - index_start, 'sec')
    return indexer

def stem_sentence_and_remove_stopwords(sentence):
    #print('before alnum', sentence)
    sentence = ''.join(ch for ch in sentence if ch.isalnum() or ch.isspace() or ch == '.,-')
    #print('after alnum', sentence)
    result = ' '.join(map(stemmer.stem, (x for x in sentence.lower().split() if x not in stop)))
    #print('after stemmer', result)
    return result

def remove_duplicate_words(q):
    r, r_set = [], set()
    for w in q.lower().split():
        if w not in r_set:
            r.append(w)
            r_set.add(w)
    return ' '.join(r)

def stem_sentence_and_remove_stopwords_ck12_keywords(sentence, is_question):
    #print('before stem', sentence)
    sentence = stem_sentence_and_remove_stopwords(sentence)
    #print('after stem', sentence)
    result = remove_duplicate_words(sentence)

    # WTF???
    keywords = []
    for keyword in ck_keywords:
        if len(keyword.split()) > 1 and keyword in result:
            keywords.append(keyword)
            break
        else:
            if keyword in result.split():
                keywords.append(keyword)
                break

    for keyword in keywords:
        if is_question:
            for word in keyword.split():
                result = result.replace(word, '"'+word+'"^1.5')
        else:
            for word in keyword.split():
                result = result.replace(word, '"'+word+'"^1.3')
    return result

def prepare_sentence(sentence,is_question=False):
    sentence = stem_sentence_and_remove_stopwords_ck12_keywords(sentence, is_question)
    return sentence

def predict_answer(indexer, question, answers, verbose=False, ca=''):
    question = prepare_sentence(question, True)
    answers = map(prepare_sentence, answers.split('\t'))

    get_score = lambda h: h[0].score if h else 0
    #print(question, answers)
    score_fun = lambda a: get_score(indexer.search(question + ' ' + a, count=10, scores=True, field='contents'))
    scores = map(score_fun, answers)
    ans = chr(np.argmax(scores) + ord('A'))

    if verbose:
        print(u'\nQ: {}\ntrue ans: {}\npred ans: {}\nA: {} ({:.3f})\nB: {} ({:.3f})\nC: {} ({:.3f})\nD: {} ({:.3f})\n'.format(question, ca, ans, *chain(*zip(answers, scores))))
    return ans, scores[0], scores[1], scores[2], scores[3]

def get_answer_from_answer_page(path, query):
    with open(path) as doc:
        content = doc.readlines()
        for x in content:
            indexer.add(path=path, contents=x)

    hits = indexer.search(query, count=10, scores=True, field='contents')
    return hits[0].score

def get_accuracy_on_train_set(indexer):
    predictor = lambda (id, q, ca, a1, a2, a3, a4): predict_answer(indexer, q, '\t'.join([a1, a2, a3, a4]), verbose=True, ca=ca)
    df = pd.read_csv(TRAIN_SET_PATH, delimiter='\t', encoding='utf-8')
    df['model_answer'], df['scoreA'], df['scoreB'], df['scoreC'], df['scoreD'] = zip(*map(predictor, df.values))
    ca = (df.model_answer == df.correctAnswer)
    df[['id', 'scoreA', 'scoreB', 'scoreC', 'scoreD']].to_csv('../data/train_lucene_scores.csv', index=False)
    return ca[ca == True].size * 1.0 / len(df)

def get_submission_from_val(indexer):
    predictor = lambda (id, q, a1, a2, a3, a4): predict_answer(indexer, q, '\t'.join([a1, a2, a3, a4]))
    df = pd.read_csv('./data/validation_set.tsv', delimiter='\t')
    df['correctAnswer'], df['scoreA'], df['scoreB'], df['scoreC'], df['scoreD'] = zip(*map(predictor, df.values))
    df[['id', 'correctAnswer']].to_csv('./data/submit.csv', index=False)
    df[['id', 'scoreA', 'scoreB', 'scoreC', 'scoreD']].to_csv('./data/val_lucene_scores.csv', index=False)

if __name__ == '__main__':
    stop = set(stopwords.words('russian'))
    stemmer = SnowballStemmer('russian')

    ck_keywords = filter(lambda w: w not in stop, codecs.open(CK_KEYWORDS_PATH, 'r', 'utf-8').read().split())
    ck_keywords = set(filter(len, map(stemmer.stem, ck_keywords)))

    lucene.initVM()
    indexer = prepare_index(DOCS_DIR, stemmer)
    #if not os.path.exists('lucene_index.pkl'):
    #    indexer = prepare_index(DOCS_DIR, stemmer)
    #    pickle.dump(indexer, open('lucene_index.pkl', 'wb'))
    #else:
    #    indexer = pickle.load(open('lucene_index.pkl', 'rb'))
    print('Accuracy on train set:', get_accuracy_on_train_set(indexer))
    print('Preparing submission...')
    get_submission_from_val(indexer)
    print('Done!')
