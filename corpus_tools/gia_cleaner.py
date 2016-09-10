# -*- coding: utf-8 -*-

import sys
from os import listdir, path
import codecs

for f in listdir(sys.argv[1]):
    if not f[-6:] == "10.txt":
        continue
    with codecs.open(path.join(sys.argv[1], f), encoding='utf-8') as f_in, \
            codecs.open(path.join(sys.argv[1], f[:-3] + "processed.txt"), "wt", encoding='utf-8') as f_out:
        write = False
        prev = ""
        for line in f_in:
            if line.lower().strip() == u"желаем успеха!" or line.lower().strip() == u"часть 1":
                write = True
            elif line.lower().strip() == u"часть 2":
                write = False
            if line.strip() == prev:
                continue
            if write:
                f_out.write(line)
                prev = line.strip()
