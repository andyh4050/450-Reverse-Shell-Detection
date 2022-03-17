
import pandas as pd
import numpy as np
import collections
import math


filenames=[1908,4696,7156,10568]
for k in filenames:
    reverse = pd.read_csv('data/benign_result'+str(k) + '.csv')
    indexes = pd.notna(reverse['CallStack'])
    reverse = reverse[indexes]
    reverse_callStack = list(reverse['CallStack'])

    reverse_callStack_result = []
    for i in reverse_callStack:
        e = i[1:][:-1]
        temp = e.split(',')
        for i in range(len(temp)):
            temp[i] = temp[i].strip("' ").strip(" '")
        reverse_callStack_result.append(temp)

    flat_reverse = [item for sublist in reverse_callStack_result for item in sublist]
    denom = len(flat_reverse)

    tf = dict(collections.Counter(flat_reverse))
    for item in tf:
        tf[item] = tf[item]/denom


    benign = pd.read_csv('data/reverse_result'+str(k) + '.csv')
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
        idf[item] = math.log(benign_denom/idf[item])

    tfidf = {}
    for item in tf:
        if item in idf:
            tfidf[item] = tf[item] * idf[item]
        else:
            tfidf[item] = tf[item]

    sorted1 = dict(sorted(tfidf.items(), key=lambda item: item[1], reverse=True))
    sorted_df = pd.DataFrame.from_dict(sorted1, orient='index', columns=['importance'])
    print(sorted_df)
    sorted_df.to_csv('result2/keyword_result_benign'+str(k) + '.csv', index = True, header=True)



