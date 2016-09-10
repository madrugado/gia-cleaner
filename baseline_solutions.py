from random import choice
import codecs
import pandas as pd
import argparse
from gensim.models import Word2Vec
from pymystem3 import Mystem
import numpy as np
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input_file", type=str, default="train.csv")
parser.add_argument("-m", "--mode", type=str, default="random", help="mode of running, e.g. random")
parser.add_argument("-w", "--word2vec_model", type=str, default=None, help="path to word2vec model file")
parser.add_argument("-o", "--output_file", type=str, default="result.csv")

args = parser.parse_args()

with codecs.open(args.input_file, encoding="utf-8") as f_in:
    questions = pd.read_csv(f_in, sep="\t")
answer_letters = ["A", "B", "C", "D"]

if args.mode == "random":
    with open(args.output_file, 'wt') as result_file:
        result_file.write('id,correctAnswer\n')
        for id_ in questions[["id"]]:
            result_file.write('{0},{1}\n'.format(id_, choice(answer_letters)))

if args.mode == "word2vec":
    pred = []
    w2v = Word2Vec.load_word2vec_format(args.word2vec_model, binary=True)
    m = Mystem()

    def get_mean_vec(phrase):
        tokens = m.analyze(phrase)
        vectors = [np.zeros((w2v.vector_size,))]
        for token in tokens:
            if "analysis" not in token:
                continue
            if token["analysis"]:
                tag = token["analysis"][0]["gr"].split(',')[0]
                if tag[-1] == "=":
                    tag = tag[:-1]
                lemma = token["analysis"][0]["lex"] + "_" + tag
                vector = np.zeros((w2v.vector_size,))
                if lemma in w2v:
                    vector = w2v[lemma]
                vectors.append(vector)
        return np.mean(vectors, axis=0)

    def get_l2_distance(v1, v2):
        return np.sqrt(np.sum(np.square(v1 - v2)))

    with open(args.output_file, 'wt') as result_file:
        result_file.write('id,correctAnswer\n')
        for q in tqdm(questions.itertuples()):
            qv = get_mean_vec(q.question)
            a1 = get_l2_distance(qv, get_mean_vec(q.answerA))
            a2 = get_l2_distance(qv, get_mean_vec(q.answerB))
            a3 = get_l2_distance(qv, get_mean_vec(q.answerC))
            a4 = get_l2_distance(qv, get_mean_vec(q.answerD))
            pred_index = np.argmax([a1, a2, a3, a4])
            result_file.write('{0},{1}\n'.format(q.id, answer_letters[pred_index]))
