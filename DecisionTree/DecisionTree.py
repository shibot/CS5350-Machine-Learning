# Author: Shibo Tang 
import math 
import copy 
import time 
import statistics 
import operator
import random

#This is node class
class Node:
    children = list()
    label = ""
    splitsOn = ""
    prediction = ""

    def __init__(self):
        self.children = list()
        self.label = ""
        self.splitsOn = ""
        self.prediction = ""


#   Majority Error Method
def Majority_Error(Data,label):
    # Create dictionary with label Probability
    Dictionary = Probability(Data,label)
    # Calculate Majority Error
    majority_error = 1 - max(Dictionary.values())

    return majority_error

#   Entropy method
def Entropy(Data,label):
    Dictionary = Probability(Data,label)
    entropy = 0.0
    for  Prob in Dictionary.values():
        if Prob == 0.0:
            continue
        entropy -= (Prob*math.log2(Prob))
    return entropy

#   Gini_Index method
def Gini_Index(Data,label):
    # Create dictionary with label Probability
    Dictionary = Probability(Data,label)
    Gini = 1.0
    for prob in Dictionary.values():
        Gini -= (prob**2)
    return Gini

#   Information gain method
def IG(Data,Columns,attribute,a_values,label,func):
    IG = func(Data,label)
    for a_value in a_values:
        Data_v = subset(Data,Columns,attribute,a_value)
        IG -= (len(Data_v)/len(Data)*func(Data_v,label))
    return IG

def Probability(Data,label):
    # Use dictionary to store data
    Dictionary = dict()
    for label in label:
        Dictionary[label] = 0
    
    for label in label:
        for values in Data:
            if values[len(values)-1] == label:
                Dictionary[label] += 1
    
    for label,total in Dictionary.items():
        if len(Data) == 0:
            Dictionary[label] = 0.0
        else:
            Dictionary[label] = total/len(Data)
   
    return Dictionary

def Predict(values,tree,Columns):
    actual = values[len(values)-1]
    current = tree

    while not current.leaf():

        decision_attr = current.name 
        attri = values[Columns.index(decision_attr)]
        current = current.branches[attri] 
    if current.name == actual:
        return True
    else:
        return False

#   Helper method to find the best split
def BestSplit(Data,Columns,Attributes,label,func):
    attr_IGs = dict()
    for attribute,a_values in Attributes.items():
        attr_IGs[attribute] = IG(Data,Columns,attribute,a_values,label,func)
    return max(attr_IGs,key=attr_IGs.get)

#   Helper method to return most common value
def Common(Data,index):

    counts = dict()
    term_set = set()

    for values in Data:
        val = values[index]
        term_set.add(val)
    
    for term in term_set:
        counts[term] = 0
    
    for term in term_set:
        for values in Data:
            if(values[index] == term):
                counts[term] += 1
    
    return max(counts,key=counts.get)

# Helper Method to find data with subset
def subset(Data,Columns,A,attriue):
    Data_v = []
    for values in Data:
        if values[Columns.index(A)] == attriue:
            Data_v.append(values)
    return Data_v


def ID3(Data,Columns,Attributes,label,func,Tree_level,current_depth):

    if(len(label) == 1):
        leaf_name = str(label.pop())
        return Node(leaf_name)

    if(len(Attributes) == 0):
        return Node(str(Common(Data,len(Columns)-1)))

    if Tree_level == current_depth:
        return Node(str(Common(Data,len(Columns)-1)))
        
    Best = BestSplit(Data,Columns,Attributes,label,func)

    root = Node(str(Best))
    
    for attriue in Attributes[Best]:
        
        Data_v = subset(Data,Columns,Best,attriue)
        
        if len(Data_v) == 0:
            root.branches[attriue] = Node(str(Common(Data,len(Columns)-1)))
        else:
            Attributes_v = copy.deepcopy(Attributes)
            Attributes_v.pop(Best)
            label_v = set()
            
            for values in Data_v:
                label_v.add(values[len(values)-1])

            root.branches[attriue] = ID3(Data_v,Columns,Attributes_v,label_v,func,Tree_level,current_depth+1)
    
    return root


# Helper method to read .csv file
def read_csv(name):
    data = []
    with open(name) as file:
        for line in file:
            data.append(line.strip().split(','))
    return data

# Helper method for test 3(b)
def Common_unknowns(Data,index):
    term_set = set()
    counts = dict()
    for values in Data:
        val = values[index]
        if val == 'unknown':
            continue
        else:
            term_set.add(val)
    for term in term_set:
        counts[term] = 0
    for term in term_set:
        for values in Data:
            if(values[index] == term):
                counts[term] += 1
    return max(counts,key=counts.get)


Columns = ['buying','maint','doors','persons','lug_boot','safety','label']
Attributes = {'buying':['vhigh','high','med','low'],'maint':['vhigh','high','med','low'],'doors':['2','3','4','5more'],'persons':['2','4','more'],'lug_boot':['small','med','big'],'safety':['low','med','high']}

def id3_with_weight(S, Attributes, MaxDepth, depth):

    if depth == MaxDepth:
        return mostCommonLabelLeaf(S)
    labelCheck = S[0]["Label"]
    allSame = True
    for s in S:
        if s["Label"] != labelCheck:
            allSame = False
            break
    if allSame:
        leaf = Node()
        leaf.prediction = labelCheck
        return leaf

    if len(Attributes) == 0:
        return mostCommonLabelLeaf(S)

    A = InfoGain(S, Attributes)

    root = Node()
    root.splitsOn = A

    for v in Attributes[A]:
        Sv = getSv(S, A, v)

        if len(Sv) == 0:
            leaf = mostCommonLabelLeaf(S, v)
            root.children.append(leaf)
        else:
            tempAttr = dict(Attributes)
            tempAttr.pop(A)
            subtree = id3_with_weight(Sv, tempAttr, MaxDepth, depth+1)
            subtree.label = v
            root.children.append(subtree)
    return root

def id3_rand_learn(S, Attributes, NumFeatures):
    labelCheck = S[0]["Label"]
    allSame = True
    for s in S:
        if s["Label"] != labelCheck:
            allSame = False
            break
    if allSame:
        leaf = Node()
        leaf.prediction = labelCheck
        return leaf

    if len(Attributes) == 0:
        return mostCommonLabelLeaf(S)
    A = InfoGain_rand(S, Attributes, NumFeatures)

    root = Node()
    root.splitsOn = A

    for v in Attributes[A]:
        Sv = getSv(S, A, v)

        if len(Sv) == 0:

            leaf = mostCommonLabelLeaf(S, v)
            root.children.append(leaf)
        else:
            tempAttr = dict(Attributes)
            tempAttr.pop(A)
            subtree = id3_rand_learn(Sv, tempAttr, NumFeatures)
            subtree.label = v
            root.children.append(subtree)

    return root


def mostCommonLabelLeaf(S, v=None):
    labels = {}
    for s in S:
        label = s["Label"]
        if label not in labels:
            labels[label] = 0.0
        labels[label] += s["Weight"]
    maxNum = 0
    maxLabel = ""
    for label, num in labels.items():
        if num > maxNum:
            maxNum = num
            maxLabel = label
    leaf = Node()
    leaf.prediction = maxLabel
    leaf.label = v
    return leaf


def getSv(S, A, v):

    return [s for s in S if s[A] == v]


def values(Attributes, A):

    return Attributes[A]


def InfoGain(S, Attributes):
    entropy = ent_calc(S)
    maxInfo = -1
    maxAttr = ""
    for A in Attributes:
        infoGain = infohelper(S, Attributes, A, entropy)
        if infoGain > maxInfo:
            maxInfo = infoGain
            maxAttr = A
    return maxAttr


def infohelper(S, Attributes, A, entropy):
    newEnt = 0.0
    for v in Attributes[A]:
        Sv = getSv(S, A, v)
        ratio = get_len(Sv)/float(get_len(S))
        ent = ent_calc(Sv)
        newEnt += ratio * ent
    return entropy - newEnt


def ent_calc(S):
    if len(S) == 0:
        return 0
    labels = {}
    for s in S:
        label = s["Label"]
        if label not in labels:
            labels[label] = 0.0
        labels[label] += s["Weight"]

    entropy = 0.0
    norm = get_len(S) 
    for (label, quant) in labels.items():
        ratio = quant/float(norm)
        entropy -= math.log((ratio), 2) * (ratio)

    return entropy

def InfoGain_rand(S, Attributes, NumFeatures):
    entropy = ent_calc(S)

    newAttrs = []
    cpyAttrs = list(Attributes)
    for _ in range(0, NumFeatures):
        if len(cpyAttrs) == 0:
            break
        rand = random.randint(0, len(cpyAttrs)-1)
        newAttrs.append(cpyAttrs[rand])
        del cpyAttrs[rand]
    maxInfo = -1
    maxAttr = ""
    for A in newAttrs:
        infoGain = infohelper(S, Attributes, A, entropy)
        if infoGain > maxInfo:
            maxInfo = infoGain
            maxAttr = A
    return maxAttr

def get_len(S):
    length = 0
    for s in S:
        length += s["Weight"]
    return length


def id3_weighted_err(Tree, S):
    wrong = 0.0
    for s in S:
        label = get_label(s, Tree)
        if label != s["Label"]:
            wrong += s["Weight"]
    return wrong


def get_label(s, Tree):
    if Tree.prediction != "":
        return Tree.prediction
    
    newTree = None
    for node in Tree.children:
        if node.label == s[Tree.splitsOn]:
            newTree = node
            break

    return get_label(s, newTree)


data1 = read_csv("./Data_sets/car_train.csv")
label = set()

for values in data1:
    label.add(values[len(values)-1])

with open('./Data_sets/car_train.csv', 'r') as train_file:
    for line in train_file:
        terms = line.strip().split(',')
        data1.append(terms)

    print("Car Start")
    time.sleep(2)
    for Tree_level in range(1,7):
        me = ID3(data1,Columns,Attributes,label,Majority_Error,Tree_level,0)
        gini = ID3(data1,Columns,Attributes,label,Gini_Index,Tree_level,0)
        entropy = ID3(data1,Columns,Attributes,label,Entropy,Tree_level,0)

        train_ME = 0
        train_GINI = 0
        train_Entropy = 0
        train_total = 0
        test_ME = 0
        test_GINI = 0
        test_Entropy = 0
        test_total = 0

        with open('./Data_sets/car_train.csv','r') as train_file:
            for line in train_file:
                values = line.strip().split(',')
                if Predict(values,me,Columns):
                    train_ME += 1
                if Predict(values,gini,Columns):
                    train_GINI += 1
                if Predict(values,entropy,Columns):
                    train_Entropy += 1
                train_total += 1

        with open('./Data_sets/car_test.csv','r') as test_file:
            for line in test_file:
                values = line.strip().split(',')
                if Predict(values,me,Columns):
                    test_ME += 1
                if Predict(values,entropy,Columns):
                    test_Entropy +=1
                if Predict(values,gini,Columns):
                    test_GINI += 1
                test_total += 1

        train_error_ME = round(100*(1-(train_ME/train_total)),2)
        train_error_EN = round(100*(1-(train_Entropy/train_total)),2)
        train_error_GINI = round(100*(1-(train_GINI/train_total)),2)
        test_error_ME = round(100*(1-(test_ME/test_total)),2)
        test_error_EN = round(100*(1-(test_Entropy/test_total)),2)
        test_error_GINI = round(100*(1-(test_GINI/test_total)),2)
        print("Tree Level: " + str(Tree_level) )
        print("train_error_ME  "+ str(train_error_ME)+"%  "+ "train_error_EN  "+str(train_error_EN)+"%  "+"train_error_GINI  "+str(train_error_GINI)+"%")
        print("test_error_ME  "+ str(test_error_ME)+"%  "+ " test_error_EN "+str( test_error_EN)+"%  "+" test_error_GINI  "+str( test_error_GINI)+"%")
        print("")
        
    print("Car end")
    time.sleep(2)
    print("bank with unknown start")
    training_data = []
    testing_data = []

    Columns = ['age','job','marital','education','default','balance',
                'housing','loan','contact','day','month','duration',
                'campaign','pdays','previous','poutcome','y']

    Attributes = {'age':['high','low'],'job':['admin.','unknown','unemployed','management','housemaid','entrepreneur','student','blue-collar','self-employed','retired',
            'technician','services'],'marital':['married','divorced','single'],'education':['unknown','secondary','primary','tertiary'],'default':['yes','no'],
        'balance':['high','low'],'housing':['yes','no'],'loan':['yes','no'],'contact':['unknown','telephone','cellular'],'day':['high','low'],
        'month':['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec'],'duration':['high','low'],'campaign':['high','low'],
        'pdays':['high','low'],'previous':['high','low'],'poutcome':['unknown','other','failure','success']}

    label = {'yes','no'}

    with open('./Data_sets/bank_train.csv', 'r') as train_file:
        for line in train_file:
            terms = line.strip().split(',')
            training_data.append(terms)

    with open('./Data_sets/bank_test.csv', 'r') as test_file:
        for line in test_file:
            terms = line.strip().split(',')
            testing_data.append(terms)

    medians = {'age':0.0,'balance':0.0,'day':0.0,'duration':0.0,'campaign':0.0,'pdays':0.0,'previous':0.0}

    for attr in medians.keys():
        s_attr = []
        for values in training_data:
            s_attr.append(float(values[Columns.index(attr)]))
        medians[attr] = statistics.median(s_attr)


    for attr,median in medians.items():
        for values in training_data:
            s_attri = float(values[Columns.index(attr)])
            if s_attri < median:
                values[Columns.index(attr)] = 'low'
            else:
                values[Columns.index(attr)] = 'high'

        for values in testing_data:
            s_attri = float(values[Columns.index(attr)])
            if s_attri < median:
                values[Columns.index(attr)] = 'low'
            else:
                values[Columns.index(attr)] = 'high'



    for Tree_level in range(1,17):
        me = ID3(training_data,Columns,Attributes,label,Majority_Error,Tree_level,0)
        gini = ID3(training_data,Columns,Attributes,label,Gini_Index,Tree_level,0)
        entropy = ID3(training_data,Columns,Attributes,label,Entropy,Tree_level,0)

        train_ME = 0
        train_GINI = 0
        train_Entropy = 0
        train_total = 0
        test_ME = 0
        test_GINI = 0
        test_Entropy = 0
        test_total = 0

        for values in training_data:
            if Predict(values,me,Columns):
                train_ME += 1
            if Predict(values,gini,Columns):
                train_GINI += 1
            if Predict(values,entropy,Columns):
                train_Entropy += 1
            train_total += 1

        for values in testing_data:
            if Predict(values,me,Columns):
                test_ME += 1
            if Predict(values,gini,Columns):
                test_GINI += 1
            if Predict(values,entropy,Columns):
                test_Entropy += 1
            test_total += 1

        train_error_ME = round(100*(1-(train_ME/train_total)),2)
        train_error_EN = round(100*(1-(train_Entropy/train_total)),2)
        train_error_GINI = round(100*(1-(train_GINI/train_total)),2)
        test_error_ME = round(100*(1-(test_ME/test_total)),2)
        test_error_EN = round(100*(1-(test_Entropy/test_total)),2)
        test_error_GINI = round(100*(1-(test_GINI/test_total)),2)
        
        print("Tree Level: " + str(Tree_level) )
        print("train_error_ME  "+ str(train_error_ME)+"%  "+ "train_error_EN  "+str(train_error_EN)+"%  "+"train_error_GINI  "+str(train_error_GINI)+"%")
        print("test_error_ME  "+ str(test_error_ME)+"%  "+ " test_error_EN "+str( test_error_EN)+"%  "+" test_error_GINI  "+str( test_error_GINI)+"%")
        print("")
        

    print("bank with unknown end")
    time.sleep(2)
    print("bank without unknown start")
    mostcommon_training_data = dict()
    mostcommon_testing_data = dict()

    for attribute in Columns:
        mostcommon_training_data[attribute] = Common_unknowns(training_data,Columns.index(attribute))
        mostcommon_testing_data[attribute] = Common_unknowns(testing_data,Columns.index(attribute))

    for values in training_data:
        for index in range(0,len(Columns)):
            if values[index] == 'unknown':
                values[index] = mostcommon_training_data[Columns[index]]

    for values in testing_data:
        for index in range(0,len(Columns)):
            if values[index] == 'unknown':
                values[index] = mostcommon_testing_data[Columns[index]] 


    for Tree_level in range(1,17):
        me = ID3(training_data,Columns,Attributes,label,Majority_Error,Tree_level,0)
        gini = ID3(training_data,Columns,Attributes,label,Gini_Index,Tree_level,0)
        entropy = ID3(training_data,Columns,Attributes,label,Entropy,Tree_level,0)

        train_ME = 0
        train_GINI = 0
        train_Entropy = 0
        train_total = 0
        test_ME = 0
        test_GINI = 0
        test_Entropy = 0
        test_total = 0

        for values in training_data:
            if Predict(values,me,Columns):
                train_ME += 1
            if Predict(values,gini,Columns):
                train_GINI += 1
            if Predict(values,entropy,Columns):
                train_Entropy += 1
            train_total += 1

        for values in testing_data:
            if Predict(values,entropy,Columns):
                test_Entropy += 1
            if Predict(values,me,Columns):
                test_ME += 1
            if Predict(values,gini,Columns):
                test_GINI += 1
            test_total += 1

        train_error_ME = round(100*(1-(train_ME/train_total)),2)
        train_error_EN = round(100*(1-(train_Entropy/train_total)),2)
        train_error_GINI = round(100*(1-(train_GINI/train_total)),2)
        test_error_ME = round(100*(1-(test_ME/test_total)),2)
        test_error_EN = round(100*(1-(test_Entropy/test_total)),2)
        test_error_GINI = round(100*(1-(test_GINI/test_total)),2)
        
        print("Tree Level: " + str(Tree_level) )
        print("train_error_ME  "+ str(train_error_ME)+"%  "+ "train_error_EN  "+str(train_error_EN)+"%  "+"train_error_GINI  "+str(train_error_GINI)+"%")
        print("test_error_ME  "+ str(test_error_ME)+"%  "+ " test_error_EN "+str( test_error_EN)+"%  "+" test_error_GINI  "+str( test_error_GINI)+"%")
        print("")








