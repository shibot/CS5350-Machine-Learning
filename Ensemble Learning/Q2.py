# Author: Shibo Tang 
from part2 import *
import random
import numpy



def calc_median(arr):
    n = len(arr)
    if n < 1:
            return None
    if n % 2 == 1:
            return sorted(arr)[n//2]
    else:
            return sum(sorted(arr)[n//2-1:n//2+1])/2.0

def reset_weights(S):
    weight = 1/float(len(S))
    for s in S:
        s["Weight"] = weight

dataset = "bank"
attrFile = open(dataset + "/data-desc.txt")
attrFile.readline()
attrFile.readline()
labels = "".join(attrFile.readline().split()).split(',')
attrFile.readline()
attrFile.readline()
attrFile.readline()
Attributes = {}
attrList = []
line = attrFile.readline()
while line != "\n":
    splitLine = line.split(':')
    attr = splitLine[0]
    attrList.append(attr)
    attrVals = "".join(splitLine[1].split()).split(',')
    Attributes[attr] = attrVals
    line = attrFile.readline()

attrList.append("Label")

numericList = [A for A in attrList if A in Attributes and Attributes[A][0] == "(numeric)"]

S_train = []
numericalLists = {}
with open(dataset + "/train.csv") as f:
    for line in f:
        i = 0
        example = {}
        for attr in line.strip().split(','):
            attrName = attrList[i]
            if attrName in numericList:
                if attrName not in numericalLists:
                    numericalLists[attrName] = []
                numericalLists[attrName].append(float(attr))
            example[attrList[i]] = attr
            i += 1
        S_train.append(example)

medianList = {}
for name, arr in numericalLists.items():
    medianList[name] = calc_median(arr)

for i in S_train:
    for attr in numericList:
        if i[attr] >= numericalLists[attr]:
            i[attr] = "1"
        elif i[attr] < numericalLists[attr]:
            i[attr] = "-1"
    i["Weight"] = 1/float(len(S_train))

S_test = []
with open(dataset + "/test.csv") as f:
    for line in f:
        i = 0
        example = {}
        for attr in line.strip().split(','):
            name = attrList[i]
            if name in numericList:
                val = float(attr)
                if val >= numericalLists[name]:
                    attr = "1"
                elif val < numericalLists[name]:
                    attr = "-1"
            example[name] = attr
            example["Weight"] = 1/5000.0 
            i += 1
        S_test.append(example)

for attr in numericList:
    Attributes[attr] = ["-1", "1"]

print "AdaBoost"

for T in range(1, 1000, 10):
    hypothesis = AdaBoost(S_train, Attributes, T)
    err_train = AdaBoost_Test(hypothesis, S_train)
    err_test = AdaBoost_Test(hypothesis, S_test)
    print str(T-1) + "\t" + str(err_train) + "\t" + str(err_test)
    reset_weights(S_train)

hypothesis = AdaBoost_print(S_train, S_test, Attributes, 1000)


print "Baggedtree"

for T in range(1, 1000, 10):
    hypothesis = Bagging_Train(S_train, Attributes, T)
    err_train = Bagging_Test(hypothesis, S_train)
    err_test = Bagging_Test(hypothesis, S_test)
    print "T = " + str(T-1) + ": " + str(err_train) + ", " + str(err_test)

    reset_weights(S_train)


print "Bias and Variance"

predictors = []
for _ in range(0, 100):
    copy_S = list(S_train)
    new_S = []
    for i in range(0, 1000):
        rand = random.randint(0, len(copy_S) - 1)
        new_S.append(copy_S[rand])
        del copy_S[rand]
    predictor = Bagging_Train(new_S, Attributes, 1000)
    predictors.append(predictor)

total_single_bias = 0.0
total_single_variance = 0.0
for s in S_test:
    avg = 0.0
    predictions = []
    for p in predictors:
        label = get_label(s, p[0][0])
        val = 1 if label == "yes" else -1
        avg += val
        predictions.append(val)
    avg /= len(predictors)
    label_num = 1 if s["Label"] == "yes" else -1

    bias = pow(label_num - avg, 2)
    total_single_bias += bias

    variance = numpy.var(predictions)
    total_single_variance += variance

single_bias = total_single_bias/len(S_test)
single_variance = total_single_variance/len(S_test)

print "Single bias: " + str(single_bias)
print "Single variance: " + str(single_variance)

total_mass_bias = 0.0
total_mass_variance = 0.0
for s in S_test:
    T += 1
    avg = 0.0
    predictions = []
    for p in predictors:
        val = get_bag_label(p, s) / float(len(p[0]))
        avg += val
        predictions.append(val)
    avg /= len(predictors)
    label_num = 1 if s["Label"] == "yes" else -1

    bias = pow(label_num - avg, 2)
    total_mass_bias += bias

    variance = numpy.var(predictions)
    total_mass_variance += variance

mass_bias = total_mass_bias/len(S_test)
mass_variance = total_mass_variance/len(S_test)

print "Mass bias: " + str(mass_bias)
print "Mass variance: " + str(mass_variance)


## Generates Data for 2d ##
print "Bias and Variance for (d)"

for T in range(1, 1000, 10):
    hypothesis = Random_Forest_Train(S_train, Attributes, T, 2)
    err_train = Random_Forest_Test(hypothesis, S_train)
    err_test = Random_Forest_Test(hypothesis, S_test)
    f.write(str(T-1) + "\t" + str(err_train) + "\t" + str(err_test) + "\n")
    reset_weights(S_train)

for T in range(1, 1000, 10):
    hypothesis = Random_Forest_Train(S_train, Attributes, T, 4)
    err_train = Random_Forest_Test(hypothesis, S_train)
    err_test = Random_Forest_Test(hypothesis, S_test)
    f.write(str(T-1) + "\t" + str(err_train) + "\t" + str(err_test) + "\n")
    reset_weights(S_train)

for T in range(1, 1000, 10):
    hypothesis = Random_Forest_Train(S_train, Attributes, T, 6)
    err_train = Random_Forest_Test(hypothesis, S_train)
    err_test = Random_Forest_Test(hypothesis, S_test)
    f.write(str(T-1) + "\t" + str(err_train) + "\t" + str(err_test) + "\n")
    reset_weights(S_train)

print "Bias and Variance for (e)"

single_predictions = len(S_test)*[None]
mass_predictions = len(S_test)*[None]

for s in range(0, len(S_test)):
    single_predictions[s] = []
    mass_predictions[s] = []

copy_S = list(S_train)
for i in range(0, 100):
    random.shuffle(copy_S)
    new_S = copy_S[:1000]
    predictor = Random_Forest_Train(new_S, Attributes, 1000, 4)

    for s in range(0, len(S_test)):
        label = get_label(S_test[s], predictor[0][0])
        val = 1 if label == "yes" else -1
        single_predictions[s].append(val)

    for s in range(0, len(S_test)):
        val = get_bag_label(predictor, S_test[s])/float(len(predictor[0]))
        mass_predictions[s].append(val)

    print "predictor " + str(i) + " done"
    
total_single_bias = 0.0
total_single_variance = 0.0
for s in range(0, len(S_test)):
    label_num = 1 if S_test[s]["Label"] == "yes" else -1
    avg_prediction = sum(single_predictions[s])/len(single_predictions[s])
    bias = pow(label_num - avg_prediction, 2)
    total_single_bias += bias

    total_single_variance += numpy.var(single_predictions[s])

single_bias = total_single_bias/100
single_variance = total_single_variance/100

print "Single Variance: " + str(single_variance)
print "Single Bias: " + str(single_bias)

total_mass_bias = 0.0
total_mass_variance = 0.0
for s in range(0, len(S_test)):
    label_num = 1 if S_test[s]["Label"] == "yes" else -1
    avg_prediction = sum(mass_predictions[s])/len(mass_predictions[s])
    bias = pow(label_num - avg_prediction, 2)
    total_mass_bias += bias

    total_mass_variance += numpy.var(mass_predictions[s])

mass_bias = total_mass_bias/100
mass_variance = total_mass_variance/100

print "Mass Variance: " + str(mass_variance)
print "Mass Bias: " + str(mass_bias)