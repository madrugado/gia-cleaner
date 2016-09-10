import argparse
import pandas as pd
import codecs

parser = argparse.ArgumentParser()
parser.add_argument("-g", "--ground-truth", type=str, help="path to file with correct answers", default="train.csv")
parser.add_argument("-t", "--test-file", type=str, help="path to file with results to test", default="result.csv")
parser.add_argument("-q", "--quiet", action="store_true", help="be quiet, print only final score")
args = parser.parse_args()

with codecs.open(args.ground_truth, encoding="utf-8") as f_in:
    questions = pd.read_csv(f_in, sep="\t")
true_answers = {q.id: q.correctAnswer for q in questions.itertuples()}

with codecs.open(args.test_file, encoding="utf-8") as f_in:
    test = pd.read_csv(f_in)
test_answers = {q.id: q.correctAnswer for q in test.itertuples()}

total_used = 0
rightly_answered = 0.
unknown = 0
for q_id in test_answers.keys():
    if q_id not in true_answers:
        unknown += 1
        continue

    if test_answers[q_id] == true_answers[q_id]:
        rightly_answered += 1

    total_used += 1

if args.quiet:
    print rightly_answered/total_used
else:
    print "Total questions in ground truth:\t%d" % len(true_answers)
    print "Total used questions in test:\t%d" % total_used
    print "Total rightly answered questions:\t%d" % rightly_answered
    print "Unknown questions number in test:\t%d" % unknown
    print "Accuracy (rightly / total used):\t%.2f" % (rightly_answered/total_used)