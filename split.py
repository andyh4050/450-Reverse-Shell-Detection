import pandas as pd
import numpy as np


reverse = pd.read_csv('reverse_result2.csv')
benign = pd.read_csv('benign_result2.csv')

r = np.random.permutation(reverse.shape[0])
b = np.random.permutation(benign.shape[0])

r_len = int(len(r)/10)
b_len = int(len(b)/10)

print('---start---')
for i in range(9):
    r_ind = r[i* r_len: (i+1) * r_len]
    reverse_temp = reverse.iloc[r_ind,:]
    b_ind = b[i* b_len: (i+1) * b_len]
    benign_temo = benign.iloc[b_ind,:]
    reverse_temp.to_csv('data/reverse_result2' + str(i) + '.csv')
    benign_temo.to_csv('data/benign_result2' + str(i) + '.csv')
    result_temp = pd.concat([benign_temo, reverse_temp], ignore_index=True)
    result_temp.to_csv('data/result2' + str(i) + '.csv')
i += 1
reverse_temp = reverse.iloc[r[i* r_len: ],:]
benign_temo = benign.iloc[b[i* b_len: ],:]
reverse_temp.to_csv('data/reverse_result2' + str(i) + '.csv')
benign_temo.to_csv('data/benign_result2' + str(i) + '.csv')
result_temp = pd.concat([benign_temo, reverse_temp], ignore_index=True)
result_temp.to_csv('data/result2' + str(i) + '.csv')