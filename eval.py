#!/usr/bin/env python
__author__ = 'arenduchintala'
import sys
import codecs
from optparse import OptionParser
import fileinput

reload(sys)
sys.setdefaultencoding('utf-8')
sys.stdin = codecs.getreader('utf-8')(sys.stdin)
sys.stdout = codecs.getwriter('utf-8')(sys.stdout)
sys.stdout.encoding = 'utf-8'

if __name__ == '__main__':

    tp = {}
    fp = {}
    fn = {}
    ls = set([])
    acc = 0
    tot = 0
    for l in fileinput.input():
        label, prediction = l.strip().split()
        ls.add(label)
        tot += 1.0
        if label == prediction:
            tp[label] = tp.get(label, 0) + 1.0
            acc += 1.0
        else:
            fp[prediction] = fp.get(prediction, 0) + 1.0
            fn[label] = fn.get(label, 0) + 1.0
    acc = acc / tot
    ave_prec = 0.0
    ave_recall = 0.0
    for l in ls:
        if (tp.get(l, 0) + fp.get(l, 0)) == 0:
            p = 0
        else:
            p = tp.get(l, 0) / (tp.get(l, 0) + fp.get(l, 0))

        if (tp.get(l, 0) + fn.get(l, 0)) == 0:
            r = 0
        else:
            r = tp.get(l, 0) / (tp.get(l, 0) + fn.get(l, 0))

        sys.stderr.write(str(l) + ',' + str(round(p, 2)) + ',' + str(round(r, 2)) + '\n')
        ave_prec += p
        ave_recall += r
    ave_prec = ave_prec / len(ls)
    ave_recall = ave_recall / len(ls)
    sys.stderr.write('ave:' + str(round(ave_prec, 4)) + ',' + str(round(ave_recall, 4)) + '\n')
    print round(acc, 4),'&', round(ave_prec, 4),'&', round(ave_recall, 4)
