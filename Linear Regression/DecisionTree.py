# Author: Shibo Tang 
import math 
import copy 
import time 
import random

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








