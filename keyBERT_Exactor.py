#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 11:48:30 2022

@author: robinluo
"""

import pandas as pd
import numpy as np
from keybert import KeyBERT
import collections
import math

DOT = 'dot'
COLON = 'colon'


result = pd.read_csv('450 ETW Data/data/result7156.csv')

indexs = pd.notna(result['CallStack'])

result = result[indexs]
callStack = list(result['CallStack'])

callStack_result = []
for i in callStack:
    e = i[1:][:-1]
    temp = e.split(',')
    for i in range(len(temp)):
        temp[i] = temp[i].strip("' ").strip(" '")
    callStack_result.append(temp)

callStack_result = np.array(callStack_result)
CallStack = callStack_result[result['processID'] == 7156]

CallStack2 = callStack_result[result['processID'] != 7156]


CallStack = str(list(CallStack))
CallStack = CallStack.replace('.', DOT)
CallStack = CallStack.replace(':', COLON)


CallStack2 = str(list(CallStack2))
CallStack2 = CallStack2.replace('.', DOT)
CallStack2 = CallStack2.replace(':', COLON)



kw_model = KeyBERT()

keywords = kw_model.extract_keywords([CallStack, CallStack2], keyphrase_ngram_range=(1,1),stop_words = ['Windows','System32','system32', 'dtype'],top_n=10000)




tf = {}



for i in keywords[0]:
    keyword = i[0]
    keyword = keyword.replace(DOT,'.')
    keyword = keyword.replace(COLON,":")
    tf[keyword] = i[1]

benign = pd.read_csv('450 ETW Data/data/benign_result7156.csv')
benign_indexes = pd.notna(benign['CallStack'])
benign = benign[benign_indexes]
benign_callStack = list(benign['CallStack'])

benign_callStack_result = []
for i in benign_callStack:
    e = i[1:][:-1]
    temp = e.split(',')
    for i in range(len(temp)):
        temp[i] = temp[i].strip("' ").strip(" '")
    benign_callStack_result.append(temp)

flat_benign = [item for sublist in benign_callStack_result for item in sublist]
benign_denom = len(flat_benign)

idf = dict(collections.Counter(flat_benign))
for item in idf:
    idf[item] = math.log(benign_denom/(1+idf[item])) + 1

tfidf = {}
for item in tf:
    if item in idf:
        tfidf[item] = tf[item] * idf[item]
    else:
        tfidf[item] = tf[item]




sorted1 = dict(sorted(tfidf.items(), key=lambda item: item[1], reverse=True))
sorted_df = pd.DataFrame.from_dict(sorted1, orient='index', columns=['importance'] )
sorted_df.to_csv('result/keyword_result_reverse7156.csv', index = True, header=True)





tf = {}

for i in keywords[1]:
    keyword = i[0]
    keyword = keyword.replace(DOT,'.')
    keyword = keyword.replace(COLON,":")
    tf[keyword] = i[1]
    
reverse = pd.read_csv('450 ETW Data/data/reverse_result7156.csv')
reverse_indexes = pd.notna(reverse['CallStack'])
reverse = reverse[reverse_indexes]
reverse_callStack = list(reverse['CallStack'])

reverse_callStack_result = []
for i in reverse_callStack:
    e = i[1:][:-1]
    temp = e.split(',')
    for i in range(len(temp)):
        temp[i] = temp[i].strip("' ").strip(" '")
    reverse_callStack_result.append(temp)

flat_reverse = [item for sublist in reverse_callStack_result for item in sublist]
reverse_denom = len(flat_reverse)

idf = dict(collections.Counter(flat_reverse))
for item in idf:
    idf[item] = math.log(reverse_denom/idf[item])

tfidf2 = {}
for item in tf:
    if item in idf:
        tfidf2[item] = tf[item] * idf[item]
    else:
        tfidf2[item] = tf[item]



sorted2 = dict(sorted(tfidf2.items(), key=lambda item: item[1], reverse=True))
sorted_df2 = pd.DataFrame.from_dict(sorted2, orient='index', columns=['importance'])
print(sorted_df2)
sorted_df2.to_csv('result/keyword_result_benign7156.csv', index = True, header=True)




