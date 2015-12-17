__author__ = 'arenduchintala'
import sys
import codecs
from optparse import OptionParser
import load_data
import math

reload(sys)
sys.setdefaultencoding('utf-8')
sys.stdin = codecs.getreader('utf-8')(sys.stdin)
sys.stdout = codecs.getwriter('utf-8')(sys.stdout)
sys.stdout.encoding = 'utf-8'


def get_value_counts(data, instance_ids, attribute):
    assert isinstance(attribute, int)
    values = set([data[i][attribute] for i in instance_ids])
    values_count = dict((v, 0) for v in values)
    for i in instance_ids:
        val = data[i][attribute]
        values.add(val)
        values_count[val] = values_count.get(val, 0) + 1
    return values_count


def get_probs(data, instance_ids, attribute):
    assert isinstance(attribute, int)
    values_count = get_value_counts(data, instance_ids, attribute)
    values_prob = dict((v, float(vc) / float(len(instance_ids))) for v, vc in values_count.iteritems())
    return values_prob


def get_attribute_instance_subsets(data, instance_ids, attribute):
    assert isinstance(attribute, int)
    values = set([data[i][attribute] for i in instance_ids])
    values_subsets = dict((v, []) for v in values)
    for i in instance_ids:
        val = data[i][attribute]
        values.add(val)
        values_subsets[val] = values_subsets.get(val, []) + [i]
    return values_subsets


if __name__ == '__main__':
    opt = OptionParser()
    # insert options here
    opt.add_option('-d', dest='data_file', default='')
    opt.add_option('-r', dest='split', default='')
    (options, _) = opt.parse_args()
    if options.data_file == '' or options.split == '':
        sys.stderr.write(
            'Usage: python naive_bayes.py -d [c/i/m1/m2/m3] -r [train test ratio]\n')
        exit(-1)
    else:
        pass
    if options.data_file == 'c':
        test_data, training_data = load_data.load_congress_data(float(options.split))
    elif options.data_file == 'i':
        test_data, training_data = load_data.load_iris(float(options.split))
    elif options.data_file == 'm1':
        test_data, training_data = load_data.load_monks(1)
    elif options.data_file == 'm2':
        test_data, training_data = load_data.load_monks(2)
    elif options.data_file == 'm3':
        test_data, training_data = load_data.load_monks(3)
    else:
        sys.stderr.write('Usage: python naive_bayes.py -d [c/i/m1/m2/m3] -r [train test ratio]\n')
        exit(-1)

    d = range(0, len(training_data))

    attr_counts = {}
    label_counts = {}
    for d in training_data:
        for f_name, f_val in enumerate(d):
            if f_name == 0:
                label_counts[f_val] = label_counts.get(f_val, 1.0) + 1.0
            else:
                f_name = int(f_name)
                label_val = int(d[0])
                attr_counts[(f_name, f_val, label_val)] = attr_counts.get((f_name, f_val, label_val), 1.0) + 1.0
    attr_probs = {}
    for k, attr_count in attr_counts.iteritems():
        f_name = k[0]
        f_val = k[1]
        label_val = k[2]
        attr_probs[k] = attr_count / label_counts[label_val]

    for t in test_data:
        tl = None
        l_given_f = []
        for l, lp in label_counts.iteritems():
            l = int(l)
            lf = math.log(lp)
            for f_name, f_val in enumerate(t):
                f_name = int(f_name)
                if f_name == 0:
                    tl = f_val
                else:
                    lf += math.log(attr_probs.get((f_name, f_val, l), 1e-8))
            l_given_f.append((lf, l))
        pred_prob, pred_label = max(l_given_f)
        print int(tl), int(pred_label)









