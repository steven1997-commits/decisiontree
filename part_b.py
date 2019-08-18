import tree
import random
import math
import matplotlib.pyplot as plt
from anytree import Node as drawNode
from anytree import RenderTree
from anytree.dotexport import RenderTreeGraph
from anytree.exporter import DotExporter

def read_and_shuffle(name):
    A = tree.read_data(name)
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
        t = tree.init_tree(train)
        t.train_tree(depth)
        acc = tree.get_prediction_accuracy(t,valid)
        acc_total += acc
        acc_train += tree.get_prediction_accuracy(t,train)
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
    res, raw = cv(5,20,data)
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

A = read_and_shuffle("occupancy_A.csv")
best = cross_validate(A, "plot_part_b.png")
print('best depth: ',best)
t = tree.init_tree(A)
t.train_tree(best)
n = t.get_plot_node()
DotExporter(n).to_dotfile("t2.dot")
