# Author: Shibo Tang 
import numpy as np
import math
import random

def add(x, y):
    output = [0]*len(x)
    for i in range(len(x)):
        output[i] = x[i] + y[i]
    return output

def gradient(w, x, y):
    dj = [0]*len(x[0])
    for i in range(0, len(y)):
        c = -(y[i] - np.dot(w, x[i]))
        dj = add(dj, [xi*c for xi in x[i]])
    return dj

def cost(w, x, y):
    j = 0.0
    for i in range(0, len(y)):
        j += 0.5*(y[i]-np.dot(w, x[i]))*(y[i]-np.dot(w, x[i]))
    return j

def bgd(r, x, y):
    err = 1.0
    t = 0
    w = [0.0]*len(x[0])
    list1 = [cost(w, x, y)]
    list2 = []
    while 1e4 >= err > 1e-6:
        dj = gradient(w, x, y)
        change = [0.0]*len(w)
        for i in range(len(w)):
            change[i] = dj[i]*(-1)*r
            w[i] = w[i] + change[i]
        list1.append(cost(w, x, y))
        err = np.linalg.norm(change)
        list2.append(err)
        t += 1
    if err > 1e4:
        output = bgd(r/2, x, y)
    else:
        output = {'cost': list1, 'error': list2, 'weight': w, 'step': t, 'r': r}
    return output


def sgd(r, x, y):
    w = [0.0]*len(x[0])
    err = 1.0
    t = 0
    k = 0
    order = list(range(len(x)))
    random.shuffle(order)
    list1 = [cost(w, x, y)]
    list2 = []
    while 1e4 >= err > 1e-6:
        i = order[k]
        change = [0.0]*len(w)
        wtx = np.dot(w, x[i])
        for j in range(len(w)):
            change[j] = r*x[i][j]*(y[i]-wtx)
            w[j] = w[j] + change[j]
        list1.append(cost(w, x, y))
        err = np.linalg.norm(change)
        list2.append(err)
        t += 1
        i += 1
        if k == len(x):
            k = 0
            random.shuffle(order)
    if err > 1e4:
        output = sgd(r/2, x, y)
    else:
        output = {'cost': list1, 'error': list2, 'weight': w, 'step': t, 'r': r}
    return output

def read_csv(csvfile):
    data = {'x': [], 'y': []}
    import csv
    with open(csvfile, 'r') as f:
        f_test = csv.reader(f)
        for row in f_test:
            new_row = []
            for x in row:
                new_row.append(float(x))
            data['x'].append(new_row[:-1] + [1])
            data['y'].append(new_row[-1])
    return data

def analytical(x, y):
    axt = np.array(x)
    ax = axt.transpose()
    w1 = np.dot(ax, axt)
    w1 = np.linalg.inv(w1)
    w1 = np.dot(w1, ax)
    w1 = np.dot(w1, y)
    return w1
