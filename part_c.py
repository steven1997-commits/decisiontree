import random
import math
import matplotlib.pyplot as plt
from anytree import Node as drawNode
from anytree import RenderTree
from anytree.dotexport import RenderTreeGraph
from anytree.exporter import DotExporter

import csv
import collections

feature_names = ['Temperature','Humidity','CO2','HumidityRatio','Occupancy']

def read_data(id):
    data = []
    with open(id) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count > 0:
                line = []
                for cat in row:
                    line.append(float(cat))
                data.append(line)
            line_count += 1
    return data

def print_arry(arr, strt, end):
    for i in range(strt, end):
        print(arr[i])

#takes in a Counter object for the distribution
def entropy(dist):
    count = 0
    for f in dist:
        if(dist[f] == float(0)):
            if(f == 'neg'):
                return float(0)
            elif(f == 'pos'):
                return float(1)
        res = dist[f] * math.log(dist[f],2)
        count += res
    return -count

#i = the arr[i] that we find are going to find
#the split points for
def find_splits(arr, i):
    splits = []
    #more than is total - less_than
    total = len(arr)
    #sort in increasing order
    lst = sorted(arr, key=lambda x: x[i])
    less_than = 0
    less_neg = 0
    prev = lst[0][4]
    for j in range(1,len(lst)):
        less_than += 1
        sign = lst[j][4]
        if (int(sign) == 0):
            less_neg += 1
        if ((not sign == prev) and (not lst[j-1][i] == lst[j][i])):
            split_point = ( lst[j-1][i] + lst[j][i] ) / 2
            more_than = total - less_than
            less_pos = less_than - less_neg
            #we will update the 'more' and 'total' categories
            #as we iterate through the rest of the array
            c = collections.Counter()
            c.update({'more': more_than, 'less': less_than}) #no update
            c.update({'less_pos': less_pos, 'less_neg': less_neg}) #no update
            c.update({'more_pos': 0, 'more_neg': 0}) #update by =
            c.update({'total_pos': 0, 'total_neg': 0}) #update by =
            splits.append([split_point, c])
        #update all data at the end
        if (j == (len(lst) - 1)):
            more_than = total - less_than
            less_pos = less_than - less_neg
            for split in splits:
                split_c = split[1]
                split_c['total_pos'] = less_pos
                split_c['total_neg'] = less_neg
                #current positives - prev positives
                split_c['more_pos'] = (less_pos - split_c['less_pos'])
                split_c['more_neg'] = (less_neg - split_c['less_neg'])
    return splits

#input is pair of (split_point, counter)
def get_info_gain(splits):
    info = []
    counter = splits[0][1]
    total = float(counter['more'] + counter['less'])
    pos_per = float(counter['total_pos']) / total
    neg_per = float(counter['total_neg']) / total
    dist = collections.Counter()
    dist.update({'pos': pos_per, 'neg': neg_per})
    e_all = entropy(dist)
    for row in splits:
        split_point = float(row[0])
        counter = row[1]
        less_pos_per = float(counter['less_pos']) / float(counter['less'])
        less_neg_per = float(counter['less_neg']) / float(counter['less'])
        more_pos_per = float(counter['more_pos']) / float(counter['more'])
        more_neg_per = float(counter['more_neg']) / float(counter['more'])
        dist_less = collections.Counter()
        dist_more = collections.Counter()
        dist_less.update({'neg': less_neg_per,'pos': less_pos_per})
        dist_more.update({'neg': more_neg_per,'pos': more_pos_per})
        e_less = entropy(dist_less)
        e_more = entropy(dist_more)
        t1 = (float(counter['less']) / total) * e_less
        t2 = (float(counter['more']) / total) * e_more
        I = e_all - t1 - t2
        info.append([split_point,I])
    return info

def get_best(info):
    lst = sorted(info, key=lambda x: x[1])
    return lst[len(lst)-1]

def choose_feature_split(data):
    num_features = len(data[0])-1
    split_points = []
    for i in range(0,num_features):
        splits = find_splits(data,i)
        if(len(splits) <= 0):
            return split_points
        #counters = build_counters(splits,data,i)
        info = get_info_gain(splits)
        best = get_best(info)
        best.append(i)
        split_points.append(best)
    split_points = sorted(split_points, key=lambda x: x[1])
    return split_points


class Node:
    data = []
    __left = None
    __right = None
    __depth = 0
    split_point = None
    __feature_num = -1
    __pos = 0
    __neg = 0
    __name = ""
    
    def __init__(self, data, name, depth):
        self.data = data
        self.__name = name
        self.__depth = depth
        for row in data:
            if(row[4] == 0):
                self.__neg += 1
            elif(row[4] == 1):
                self.__pos += 1

    def get_decision(self, data_point):
        feature_num = self.__feature_num
        if(self.__right == None and self.__left == None):
            #return majority here
            if(self.__pos > self.__neg):
                return float(1)
            else:
                return float(0)
        elif(not feature_num == -1):
            value = data_point[feature_num]
            if(value > self.split_point):
                return self.__left.get_decision(data_point)
            else:
                return self.__right.get_decision(data_point)
            
    def split_node(self):
        splits = choose_feature_split(self.data)
        if(len(splits) == 0):
            return False
        #print("length of splits: "+str(len(splits)))
        split = splits.pop()
        self.__feature_num = split[2]
        self.split_point = split[0]
        left_name = feature_names[self.__feature_num] + ': ' + ">" + str(self.split_point)
        right_name = feature_names[self.__feature_num] + ': ' + "<=" + str(self.split_point)
        left_data = []
        right_data = []
        for row in self.data:
            if(row[self.__feature_num] > self.split_point):
                left_data.append(row)
            else:
                right_data.append(row)
        self.data = []
        self.__left = Node(left_data,left_name,self.__depth + 1)
        self.__right = Node(right_data,right_name,self.__depth + 1)
        return True

    #assumes data was already given when tree was created
    def train_tree(self, max_depth=None ):
        if(max_depth is not None and max_depth <= 0):
            return
        if(self.__pos == 0 or self.__neg == 0):
            return
        res = self.split_node()
        if(res == True):
            if(max_depth is not None):
                self.__left.train_tree(max_depth - 1)
                self.__right.train_tree(max_depth - 1)
            else:
                self.__left.train_tree()
                self.__right.train_tree()

    def get_plot_node(self, prnt=None ):
        curr = drawNode(self.__name, parent=prnt)
        if(self.__left is not None):
            self.__left.get_plot_node(curr)
        if(self.__right is not None):
            self.__right.get_plot_node(curr)
        return curr

def init_tree(data):
    tree = Node(data, "root", 0)
    return tree

def get_prediction_accuracy(tree,data):
    wrong = 0
    total = 0
    for row in data:
        res = tree.get_decision(row)
        real = row[4]
        if(not real == res):
            wrong += 1
        total += 1
    return float(total - wrong) / float(total)

def plot_tree(tree, file_path):
    node = tree.get_plot_node()
    DotExporter(node).to_dotfile(file_path)

def read_and_shuffle(name):
    A = read_data(name)
    random.shuffle(A)
    return A

def split(num, arr):
    l = float(len(arr))
    size = int(math.ceil(l / num))
    pieces = []
    for i in range(0,num):
        strt = i * size
        stop = (i + 1) * size
        piece = arr[strt:stop]
        pieces.append(piece)
    return pieces

#cross validates and finds the average error
#given a single max_depth
#expects arrs as 10 sub-arrays of arr
def single_depth_cv(depth, arrs):
    acc_total = float(0)
    acc_train = float(0)
    for i in range(0,5):
        a1 = i * 2
        a2 = i * 2 + 1
        valid = arrs[a1] + arrs[a2]
        train = []
        for j in range(0,10):
            if(j is not a1 and j is not a2):
                train = train + arrs[j]
        #print(len(valid) + len(train))
        t = init_tree(train)
        t.train_tree(depth)
        acc = get_prediction_accuracy(t,valid)
        acc_total += acc
        acc_train += get_prediction_accuracy(t,train)
    return (acc_total / float(5)) , (acc_train / float(5))

def cv(strt_depth, end_depth, arr):
    arrs = split(10,arr)
    accs = []
    for i in range(strt_depth, end_depth + 1):
        val , tra = single_depth_cv(i,arrs)
        accs.append([i,val,tra])
        print("Depth, valid, train: ",i,val,tra)
    sort = sorted(accs, key=lambda x: x[1])
    return sort, accs

#assume data was already shuffled by the function read_and_shuffle
def cross_validate(data, file_path):
    res, raw = cv(10,25,data)
    best = res[-1]
    x = list(map(lambda x: x[0], raw))
    y1 = list(map(lambda y: y[1], raw))
    y2 = list(map(lambda y: y[2], raw))
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(x,y1)
    ax1.set_ylabel('Validation')
    ax2 = ax1.twinx()
    ax2.plot(x, y2, 'r-')
    ax2.set_ylabel('Training',color='r')
    for tl in ax2.get_yticklabels():
        tl.set_color('r')
    plt.savefig(file_path)
    return best[0]

A = read_and_shuffle("occupancy_B.csv")
best = cross_validate(A, "plot_part_c.png")
print('best: ', best)
t = init_tree(A)
t.train_tree(best)
n = t.get_plot_node()
DotExporter(n).to_dotfile("t3.dot")
