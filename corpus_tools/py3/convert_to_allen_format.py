# coding=utf-8
import os
import re
import collections
import math

questions_dir = 'processed_tests'
merged_questions_dir = 'merged_corpus'

hist_prefix = 'IS'
geo_prefix = 'GE'
soc_prefix = 'OB'
ans_map = {1: 'A', 2: 'B', 3: 'C', 4: 'D'}


def extract_candidate_answer(ans_line):
    return ans_line.strip().split(')', maxsplit=1)[1].strip()


def extract_answer(ans_line):
    return ans_line.strip().split('Ответ:', maxsplit=1)[1].strip()


def extract_questions(file_path):
    questions = []
    with open(os.path.join(root, f)) as q_file:
        lines = q_file.readlines()
        num_questions = math.floor(len(lines) / 8)
        for q_num in range(num_questions):
            l_idx = q_num * 8
            q = lines[l_idx].strip().split('.', maxsplit=1)[1]
            a1, a2, a3, a4 = map(extract_candidate_answer, lines[l_idx + 1: l_idx + 5])
            ans = extract_answer(lines[l_idx + 6])
            ans_letter = ans_map[int(ans)]

            print('Q: {}\nA: {}\nB: {}\nC: {}\nD: {}\nANS: {}\n'.format(q, a1, a2, a3, a4, ans))
            questions.append((q, ans_letter, a1, a2, a3, a4))
    return questions


def extract_geo_questions(file_path):
    questions = []
    with open(os.path.join(root, f)) as q_file:
        text = q_file.read()
        extracted_questions = re.findall('\d+\.((.|\n)+?)1\)\s*', text)
        extracted_candidate_answers = re.findall('\s*(1\).+\n\s*2\).+\n\s*3\).+\n\s*4\).+\n)', text)
        extracted_answers = re.findall('(Ответ: \d+\s*\n)', text)
        for i, question in enumerate(extracted_questions):
            q = question[0].strip().replace("\n", "")
            ans = extract_answer(extracted_answers[i])
            ans_letter = ans_map[int(ans)]
            candidate_answers = extracted_candidate_answers[i].split("\n")
            a1, a2, a3, a4 = map(extract_candidate_answer, candidate_answers[:-1])

            print('Q: {}\nA: {}\nB: {}\nC: {}\nD: {}\nANS: {}\n'.format(q, a1, a2, a3, a4, ans_letter))
            questions.append((q, ans_letter, a1, a2, a3, a4))
    return questions


if __name__ == '__main__':
    merged_questions = collections.defaultdict(list)
    for root, dirs, files in os.walk(questions_dir):
        for f in files:
            f_path = os.path.join(root, f)
            questions = []
            if f.startswith(hist_prefix):
                questions = extract_questions(f_path)
                prefix = hist_prefix
            elif f.startswith(soc_prefix):
                questions = extract_questions(f_path)
                prefix = soc_prefix
            elif f.startswith(geo_prefix):
                questions = extract_geo_questions(f_path)
                prefix = geo_prefix
            if questions:
                merged_questions[prefix].extend(questions)

    print('num questions per topic:')
    for topic in merged_questions:
        print(topic, len(merged_questions[topic]))

    os.makedirs(merged_questions_dir, exist_ok=True)

    # Write questions to separate topic files.
    for topic in merged_questions:
        questions_path = os.path.join(merged_questions_dir, topic + '_questions.txt')
        with open(questions_path, 'w') as q_file:
            q_file.write('id\tquestion\tcorrectAnswer\tanswerA\tanswerB\tanswerC\tanswerD\n')
            for idx, q in enumerate(merged_questions[topic]):
                q_file.write(str(idx) + '\t' + '\t'.join(q) + '\n')

    # Write all question to one file.
    questions_path = os.path.join(merged_questions_dir, 'all_questions.txt')
    idx = 0
    with open(questions_path, 'w') as q_file:
        q_file.write('id\tquestion\tcorrectAnswer\tanswerA\tanswerB\tanswerC\tanswerD\n')
        for topic in merged_questions:
            for q in merged_questions[topic]:
                q_file.write(str(idx) + '\t' + '\t'.join(q) + '\n')
                idx += 1
