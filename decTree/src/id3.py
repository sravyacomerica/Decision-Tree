#!/usr/bin/env python
# Author: Sri Sravya Tirupachur Comerica
# ID: 11259523
# Course Number: CSCE 5380 Data Mining
# Instructor: Eduardo Blanco Villar
# Implementation: Implementation of a decision tree using Entropy and Information Gain.
# Also, predicts and calculates the accuracy of a given dataset
## Example of execution: id3.py ../data/train.dat ../data/test.dat

import itertools
import logging
import sys
from math import log
from optparse import OptionParser
from typing import Any, Union

from decTree import decTree


## Reads corpus and creates the appropiate data structures:
def read_corpus(file_name):
    f = open(file_name, 'r')

    ## first line contains the list of attributes
    attr = {}
    ind = 0
    for att in f.readline().strip().split("\t"):
        attr[att] = {'ind': int(ind)}
        ind += 1

    ## the rest of the file contains the instances
    instances = []
    ind = 0
    for inst in f.readlines():
        inst = inst.strip()
        elems = inst.split("\t")
        if len(elems) < 3: continue
        instances.append({'values': map(int, elems[0:len(elems)]),
                          'class': int(elems[-1]),
                          'index': int(ind),
                          })
        ind += 1

    return attr, instances


"""
You are responsible for the implementation, but I recommend the following methods:
- generate_tree(instances): given instances, return a decision tree generated using information gain
- you probably want methods to calculate entropy, information gain, and the most useful attribute to split
- calc_accuracy(tree, instances): given a tree and instances, return the accuracy of the tree (on the instances)
- predict(tree, instance): given a tree and an instance, return the class of the instance according to the tree
"""

if __name__ == '__main__':
    usage = "usage: %prog [options] TRAINING_FILE TEST_FILE"

    parser = OptionParser(usage=usage)
    parser.add_option("-d", "--debug", action='store_true',
                      help="Turn on debug mode")

    (options, args) = parser.parse_args()
    if len(args) != 2:
        parser.error("Incorrect number of arguments")

    if options.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.CRITICAL)

    file_tr = args[0]
    file_te = args[1]
    logging.info("Training: " + file_tr)
    logging.info("Testing: " + file_te)

    attr, dataset = read_corpus(file_tr)

    """
    YOUR CODE GOES HERE
    You probably will define methods to:
    - generate a decision tree
    - predict instances and calculate accuracy

    HIGH LEVEL STEPS:
    1.- Generate a decision tree from the training instances
    2.- Print the decision tree
    3.- Predict training instances with the decision tree, calculate accuracy and print accuracy
    4.- Predict test instances with the decision tree, calculate accuracy and print accuracy
    """


    # calculates the entropy and information gain and returns two lists(left and right) based on the best splitting attribute
    def entropy_infogain(dataset):
        class_one = 0
        class_zero = 0
        for i in range(0, dataset.__len__()):
            if dataset[i][dataset[1].__len__() - 1] == 0:
                class_zero = class_zero + 1
            else:
                class_one = class_one + 1
        try:
            class_zero = (class_zero / dataset.__len__() * log(class_zero / dataset.__len__(), 2))
        except:
            class_zero = 0
        try:
            class_one = (class_one / dataset.__len__() * log(class_one / dataset.__len__(), 2))
        except:
            class_one = 0
        entropy = -(class_zero + class_one)
        total_information_gain = []
        length_instances = dataset.__len__()
        if dataset[1].__len__() > 1:
            for j in range(0, dataset[1].__len__() - 1):
                attr1_1 = 0
                attr1_0 = 0
                attr2_0 = 0
                attr2_1 = 0
                for i in range(0, length_instances):
                    if dataset[i][j] == 1:
                        if dataset[i][dataset[1].__len__() - 1] == 0:
                            attr1_0 += 1
                        elif dataset[i][dataset[1].__len__() - 1] == 1:
                            attr1_1 += 1
                    else:
                        if dataset[i][dataset[1].__len__() - 1] == 0:
                            attr2_0 += 1
                        elif dataset[i][dataset[1].__len__() - 1] == 1:
                            attr2_1 += 1
                try:
                    x = ((attr1_0 / (attr1_0 + attr1_1)) * log(attr1_0 / (attr1_0 + attr1_1), 2))
                except:
                    x = 0
                try:
                    y = ((attr1_1 / (attr1_0 + attr1_1)) * log(attr1_1 / (attr1_0 + attr1_1), 2))
                except:
                    y = 0
                try:
                    x1 = ((attr2_0 / (attr2_0 + attr2_1)) * log(attr2_0 / (attr2_0 + attr2_1), 2))
                except:
                    x1 = 0
                try:
                    y1 = ((attr2_1 / (attr2_0 + attr2_1)) * log(attr2_1 / (attr2_0 + attr2_1), 2))
                except:
                    y1 = 0

                entropy_attr1 = - (x + y)

                entropy_attr2 = - (x1 + y1)

                total = attr1_0 + attr1_1 + attr2_0 + attr2_1
                information_gain = entropy - ((attr1_0 + attr1_1) / total * entropy_attr1 + (
                        attr2_0 + attr2_1) / total * entropy_attr2)

                total_information_gain.insert(j, information_gain)
            splitting_attribute = total_information_gain.index(max(total_information_gain))
            # returns lists based on the split
            post_splitting = split_attributes(splitting_attribute, length_instances, dataset)
            return {'value': splitting_attribute, 'left_right_nodes': post_splitting}


    # splits the data set into leftsubtree and rightsubtree based on the splitting attribute passed as an argument
    def split_attributes(splitting_attribute, length_instances, instances):
        left = []
        right = []
        for i in range(0, length_instances):
            if instances[i][splitting_attribute] == 0:
                left.append(instances[i])
                index = left.__len__() - 1
                del left[index][splitting_attribute]
            else:
                right.append((instances[i]))
                index = right.__len__() - 1
                del right[index][splitting_attribute]
        return left, right


    # generates a leaf node and predicts the attribute that is highest in number
    def leaf_node(group):
        max_value = [row[-1] for row in group]
        return max(set(max_value), key=max_value.count)


    # splits a given node into root, internal and leaf nodes
    def tree_build(node, name_attrs, max_depth, min_size, depth):
        left, right = node['left_right_nodes']
        del (node['left_right_nodes'])
        if not left or not right:
            node['left'] = node['right'] = leaf_node(left + right)
            return
        if depth >= max_depth:
            node['left'], node['right'] = leaf_node(left), leaf_node(right)
            return
        if len(left) <= min_size:
            node['left'] = leaf_node(left)
        else:
            left_node = entropy_infogain(left)
            name_index = int(left_node['value'])
            left_node['name'] = name_attrs[name_index]
            left_attrs = name_attrs[:]
            del left_attrs[name_index]
            node['left'] = left_node
            tree_build(node['left'], left_attrs, max_depth, min_size, depth + 1)
        if len(right) <= min_size:
            node['right'] = leaf_node(right)

        else:
            right_node = entropy_infogain(right)
            name_index = int(right_node['value'])
            right_node['name'] = name_attrs[name_index]
            right_attrs = name_attrs[:]
            del right_attrs[name_index]
            node['right'] = right_node
            tree_build(node['right'], right_attrs, max_depth, min_size, depth + 1)


    # call the tree_build function and passes the required parameters such as depth and minimum size
    def decision_tree(instances, max_depth, min_size):
        name_attrs = ['wesley', 'romulan', 'poetry', 'honor', 'tea', 'barclay']
        root = entropy_infogain(instances)
        name_index = int(root['value'])
        root['name'] = name_attrs[name_index]
        del name_attrs[name_index]
        tree_build(root, name_attrs, max_depth, min_size, 1)
        return root


    # pretty prints tree
    def print_tree(node, depth):
        print_line = ''
        for i in range(0, depth):
            print_line += '| '

        print_line += node['name'] + ' = '

        print(print_line + '0 : ', end='')
        next_node = node['left']
        if not isinstance(next_node, int):
            print()
            print_tree(next_node, depth + 1)
        else:
            print(next_node)

        print(print_line + '1 : ', end='')
        next_node = node['right']
        if not isinstance(next_node, int):
            print()
            print_tree(next_node, depth + 1)
        else:
            print(next_node)


    # traverses through the built decision tree to predict the values of the test instances
    def traverse_tree(node, row):
        for i in range(0, row.__len__()):
            if row[i] == 0:
                if isinstance(node['left'], dict):
                    return traverse_tree(node['left'], row)
                else:
                    return node['left']
            else:
                if isinstance(node['right'], dict):
                    return traverse_tree(node['right'], row)
                else:
                    return node['right']


    # calculates the accuracy of the predicted instances with the actual instances
    def calculate_accuracy(train_set, test_set):
        for row in test_set:
            row1 = list(row)
            row1[-1] = None
            actual = [row[-1] for row in test_set]
        tree = decision_tree(train_set, 6, 6)
        print_tree(tree, 0)
        predicted_result = []
        for row in test_set:
            predict_value = traverse_tree(tree, row)
            predicted_result.append(predict_value)
        predicted = predicted_result
        right_predictions = 0
        for i in range(len(actual)):
            if actual[i] == predicted[i]:
                right_predictions = right_predictions + 1
        accuracy = (right_predictions / float(len(actual))) * 100
        return accuracy


    ## training instances
    attr_tr, instances_tr = read_corpus(file_tr)
    ## test instances
    attr_te, instances_te = read_corpus(file_te)
    train_instances = []
    test_instances = []
    # converts the list of objects into a list of training instances
    list1 = list(filter(lambda instance: "values" in instance, instances_tr))
    for instance in list1:
        train_instances.append(list(instance['values']))
    # converts the list of objects into a list of test instances
    list2 = list(filter(lambda instance: "values" in instance, instances_te))
    for instance in list2:
        test_instances.append(list(instance['values']))
    accuracy = calculate_accuracy(train_instances, test_instances)
    print("Accuracy for the given instances is (",test_instances.__len__(),"):", accuracy, "%")
