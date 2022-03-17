import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.model_selection import train_test_split


filenames=[1908]
lis = pd.DataFrame(columns=['keyword','importance'])
for i in filenames:
    print('---------------------' + str(i) + '---------------------')
    benign = pd.read_csv('result2/keyword_result_benign' + str(i) +'.csv')
    reverse = pd.read_csv('result2/keyword_result_reverse' + str(i) +'.csv')
    lists = list(benign['keyword'])
    #rint(reverse['keyword'][0].lower() in lists)
    for i in range(len(reverse)):
        if reverse['keyword'][i] not in lists:
            lis.append(reverse.iloc[i])
            print(reverse.iloc[i])


