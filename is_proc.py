# coding=utf-8
import sys
import codecs
from copy import copy

status = "question"
with codecs.open(sys.argv[1], encoding="utf-8") as f:
    q = []
    a = []
    r = []
    temp_q = []
    temp_a = []
    for line in f:
        if line.strip() == u"Ответы:":
            status = "right_answers"
            continue
        if line.strip()[:2] == "1)":
            q.append(" ".join(temp_q))
            temp_q = []
            status = "answer"
        if status == "question":
            temp_q.append(line.strip())
        if line.strip() == "" and status != "right_answers" and status != "question":
            a.append(copy(temp_a))
            temp_a = []
            status = "question"
        if status == "answer":
            temp_a.append(line.strip()[2:])
        if status == "right_answers":
            a.append(temp_a)
            if line.strip() != "":
                r.append(line.strip())

with codecs.open(sys.argv[1], "wt", encoding="utf-8") as f:
    for i in range(len(q)):
        ind = q[i].find(".")
        if ind == 1 or ind == 2:
            try:
                int(q[i][:ind])
            except ValueError:
                q[i] = str(i) + ". " + q[i]
        else:
            q[i] = str(i + 1) + ". " + q[i]
        f.write(q[i] + "\n")
        for j in range(len(a[i])):
            f.write(str(j + 1) + ") " + a[i][j] + "\n")
        f.write("\n")
        f.write(u"Ответ: " + r[i] + "\n")
        f.write("\n")
