import pandas as pd
import numpy as np
import json
import os
from os import listdir
from os.path import isfile, join
import math

mypath = '450 ETW Data/icmpsh Data/icmpsh pid 21312'
print(21312)
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]


data_total = pd.DataFrame([])

for i in onlyfiles:
    with open(mypath + '/' +i) as f:
        data = pd.DataFrame(json.loads(line) for line in f)
        data_total = pd.concat([data_total, data], ignore_index=True)




process_id_list = pd.unique(data_total['processID'])


data_total = data_total.drop(['arguments', 'EventName'], axis=1)





print('Start Splitting')
def data_split(data_temp):
    data_temp = data_temp.reset_index()
    for j in range(data_temp.shape[0]):
        try:
            data_temp['CallStack'][j] = data_temp['CallStack'][j].replace("\Windows\System32\\", '')
            data_temp['CallStack'][j] = data_temp['CallStack'][j].strip('').split('->')[:-1]
        except:
            pass
    return data_temp



data_temp= data_total
result = data_split(data_temp)

result.to_csv('450 ETW Data/data/result21312.csv')
reverse_result = result[result['processID'] == 21312]

benign_result = result[result['processID'] != 21312]
reverse_result.to_csv('450 ETW Data/data/reverse_result21312.csv')
benign_result.to_csv('450 ETW Data/data/benign_result21312.csv')
print('done')







