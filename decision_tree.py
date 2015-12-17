#!/usr/bin/env python
__author__ = 'arenduchintala'
import sys
import codecs
from optparse import OptionParser
import load_data
import numpy
import math
import pdb
from scipy import stats
from pprint import pprint

'''reload(sys)
sys.setdefaultencoding('utf-8')
sys.stdin = codecs.getreader('utf-8')(sys.stdin)
sys.stdout = codecs.getwriter('utf-8')(sys.stdout)
sys.stdout.encoding = 'utf-8'
'''


class DT_Node(object):
    def __init__(self, attribute, value, instance_ids, remaining_attrs):
        self.attribute = attribute
        self.value = value
        self.instance_ids = instance_ids
        self.remaining_attrs = remaining_attrs
        self.children = []
        self.parent = None
        self.prediction = None

    def __str__(self):
        return '|'.join([str(self.attribute), str(self.value), ','.join([str(i) for i in sorted(self.instance_ids)])])

    def size(self):
        return 1 + sum([c.size() for c in self.children])

    def get_prediction(self, instance):
        if len(self.children) == 0:
            return self.prediction
        else:
            c = [c for c in self.children if c.value == instance[c.attribute]]
            if len(c) == 0:
                return self.prediction
            else:
                return c[0].get_prediction(instance)

    def add_child(self, child):
        self.children.append(child)
        child.parent = self

    def remove_child(self, child):
        self.children = [c for c in self.children if str(c) != str(child)]

    def trace_decisions(self):
        if self.parent is None:
            return [None, None]
        else:
            return [(self.attribute, self.value)] + self.parent.trace_decisions()

    def set_prediction(self, data):
        label_counts = {}
        for i in self.instance_ids:
            label = data[i][0]
            label_counts[label] = label_counts.get(label, 0) + 1
        pred_count, pred_label = max([(c, k) for k, c in label_counts.iteritems()])
        self.prediction = pred_label

    def isTerminal(self):
        if len(self.instance_ids) == 1:
            # sys.stderr.write("1 instance left " + str(self.instance_ids[0]) + "\n")
            return True
        else:
            if len(self.remaining_attrs) == 0:
                # sys.stderr.write("no more attributes " + str(len(self.instance_ids)) + " instances\n")
                return True
            else:
                return False


class DecisionTree(object):
    def __init__(self):
        self.data = None
        self.num_attributes = None
        self.root = None

    def get_split_info(self, instance_ids, attribute):
        probs = self.get_probs(instance_ids, attribute)
        e = self.get_entropy(instance_ids, attribute)
        ps = sum([-p * math.log(p, 2) for v, p in probs.iteritems()])
        assert e == ps
        return ps

    def get_combo_counts(self, instance_ids, attributes):
        assert isinstance(attributes, list)
        assert len(attributes) == 2
        combo_counts = {}
        for i in instance_ids:
            d = self.data[i]
            da = tuple([d[attributes[0]], d[attributes[1]]])
            combo_counts[da] = combo_counts.get(da, 0) + 1
        return combo_counts

    def prune(self, n, best_attr):
        attr_counts = self.get_value_counts(n.instance_ids, best_attr)
        label_counts = self.get_value_counts(n.instance_ids, 0)
        label_probs = self.get_probs(n.instance_ids, 0)
        la_counts = self.get_combo_counts(n.instance_ids, [0, best_attr])
        chi = 0.0
        exp_c = {}
        real_c = {}
        for av in attr_counts.keys():
            for l in label_probs.keys():
                ec = attr_counts[av] * label_probs[l]
                exp_c[av] = exp_c.get(av, 0.0) + ec
                rc = la_counts.get((l, av), 0)
                real_c[av] = real_c.get(av, 0) + rc
                chi += ((rc - ec) ** 2 / ec)

        pval = 1 - stats.chi2.cdf(chi, (len(attr_counts) - 1) * (len(label_counts) - 1))
        # pprint(attr_counts)
        # pprint(label_counts)
        # pprint(label_probs)
        # print chi, pval
        return pval > 0.95

    def get_gain_ratio(self, instance_ids, attribute):
        gr = self.get_split_info(instance_ids, attribute)
        if gr > 0:
            return self.get_gain(instance_ids, attribute) / gr
        else:
            return float('inf')

    def get_gain(self, instance_ids, attribute):
        assert isinstance(attribute, int)
        values = set([self.data[i][attribute] for i in instance_ids])
        values_count = dict((v, 0) for v in values)
        values_instances = dict((v, []) for v in values)
        for i in instance_ids:
            val = self.data[i][attribute]
            values.add(val)
            values_count[val] = values_count.get(val, 0) + 1
            values_instances[val] = values_instances.get(val, []) + [i]
        label_entropy = self.get_entropy(instance_ids, 0)
        label_subsets = self.get_attribute_instance_subsets(instance_ids, 0)
        label_probs = self.get_probs(instance_ids, 0)
        lpe = 0.0
        for label_vals, label_subset in label_subsets.iteritems():
            attr_entropy = self.get_entropy(label_subset, attribute)
            lpe += label_probs[label_vals] * attr_entropy
        return label_entropy - lpe

    def get_attribute_instance_subsets(self, instance_ids, attribute):
        assert isinstance(attribute, int)
        values = set([self.data[i][attribute] for i in instance_ids])
        values_subsets = dict((v, []) for v in values)
        for i in instance_ids:
            val = self.data[i][attribute]
            values.add(val)
            values_subsets[val] = values_subsets.get(val, []) + [i]
        return values_subsets

    def get_value_counts(self, instance_ids, attribute):
        assert isinstance(attribute, int)
        values = set([self.data[i][attribute] for i in instance_ids])
        values_count = dict((v, 0) for v in values)
        for i in instance_ids:
            val = self.data[i][attribute]
            values.add(val)
            values_count[val] = values_count.get(val, 0) + 1
        return values_count

    def get_probs(self, instance_ids, attribute):
        assert isinstance(attribute, int)
        values_count = self.get_value_counts(instance_ids, attribute)
        values_prob = dict((v, float(vc) / float(len(instance_ids))) for v, vc in values_count.iteritems())
        return values_prob

    def get_entropy(self, instance_ids, attribute):
        assert isinstance(attribute, int)
        values_prob = self.get_probs(instance_ids, attribute)
        return sum([-p * math.log(p, 2) for v, p in values_prob.iteritems()])

    def predict(self, instance):
        return self.root.get_prediction(instance)

    def train(self, data, method, prune):
        self.data = data
        d = range(0, len(self.data))
        self.num_attributes = len(data[0]) - 1
        _stack = []
        root = DT_Node(None, None, d, range(1, self.num_attributes))
        _stack.append(root)
        while len(_stack) > 0:
            n = _stack.pop()

            if method == 'g':
                gains = sorted([(self.get_gain(n.instance_ids, attr), attr) for attr in n.remaining_attrs],
                               reverse=True)
            else:
                gains = sorted([(self.get_gain_ratio(n.instance_ids, attr), attr) for attr in n.remaining_attrs],
                               reverse=True)
            best_gain, best_attr = gains[0]

            instance_splits = self.get_attribute_instance_subsets(n.instance_ids, best_attr)
            # if self.prune(n, best_attr):
            # sys.stderr.write('prune\n')
            # else:
            ra = [a for a in n.remaining_attrs if a != best_attr]
            assert len(ra) < len(n.remaining_attrs)
            for split_val, split_instances in instance_splits.iteritems():
                cn = DT_Node(best_attr, split_val, split_instances, ra)
                cn.set_prediction(self.data)
                n.add_child(cn)
                if cn.isTerminal():
                    # print cn.trace_decisions()
                    pass
                else:
                    _stack.append(cn)
        self.root = root
        if prune:
            self.perform_prune()
        return self.root

    def get_leaves(self):
        leaves = []
        _stack = [self.root]
        while len(_stack) > 0:
            n = _stack.pop()
            if len(n.children) > 0:
                for c in n.children:
                    _stack.append(c)
            else:
                leaves.append(n)
        return leaves

    def perform_prune(self):
        sys.stderr.write("trying to prune\n")
        leaves = self.get_leaves()
        _stack = [l.parent for l in leaves]
        while len(_stack) > 0:
            n = _stack.pop()
            if len(n.children) > 0:
                attr = n.children[0].attribute
                if self.prune(n, attr) and n.parent is not None:
                    sys.stderr.write("pruning:" + str(n.attribute) + "," + str(n.value) + "\n")
                    p = n.parent
                    p.remove_child(n)
                    _stack.append(p)
            else:
                pass

        return True

    def test(self, data):
        for i in data:
            label = i[0]
            p = self.predict(i)
            print label, p


if __name__ == '__main__':
    opt = OptionParser()
    # insert options here
    opt.add_option('-d', dest='data_file', default='')
    opt.add_option('-m', dest='metric', default='')
    opt.add_option('-p', dest='prune', default='')
    opt.add_option('-r', dest='split', default='')
    (options, _) = opt.parse_args()
    if options.data_file == '' or options.metric == '' or options.prune == '' or options.split == '':
        sys.stderr.write(
            'Usage: python decision_tree.py -d [c/i/m1/m2/m3] -r [train test ratio] -p [true/false] -m [g/gr]\n')
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
        sys.stderr.write('Usage: python decision_tree.py -d [c/i/m1/m2/m3] -p [true/false] -m [g/gr]\n')
        exit(-1)
    num_attributes = len(training_data[0]) - 1
    dt = DecisionTree()
    dt.data = training_data
    dt.train(training_data, method=options.metric, prune=options.prune.lower() == 'true')
    sys.stderr.write("number of nodes:" + str(dt.root.size()) + '\n')
    dt.test(test_data)



