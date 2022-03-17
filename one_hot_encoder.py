import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.model_selection import train_test_split





class Preprocsss():
    def __init__(self):
        result = pd.read_csv('result.csv')
        
        
        indexs = pd.notna(result['CallStack'])
        
        result = result[indexs]
        result = result[["processID", "CallStack"]]
        self.result = result
        callStack = list(result['CallStack'])
        
        
        
        benigh_keyword = pd.read_csv('result/keyword_result_benign.csv')
        reverse_keyword = pd.read_csv('result/keyword_result_reverse.csv')
        
        benigh_keyword.sort_values(by=['importance'])
        reverse_keyword.sort_values(by=['importance'])
        
        reverse_size = int(reverse_keyword.shape[0] /10)
        benign_size = int(benigh_keyword.shape[0] /10)
        
        
        
        benigh_keyword_del = list(benigh_keyword[:benign_size]['keyword'])
        reverse_keyword_del = list(reverse_keyword[:reverse_size]['keyword'])
        
        
        
        
        callStack_result_temp = []
        y = []
        for i in callStack:
            e = i[1:][:-1]
            temp = e.split(',')
            for i in range(len(temp)):
                temp[i] = temp[i].strip("' ").strip(" '")
                
                if temp[i].lower()  in benigh_keyword_del and temp[i].lower()  in reverse_keyword_del:
                    temp[i] = ''
                
            callStack_result_temp.append(temp)

        process_id_list = pd.unique(result['processID'])
        callStack_result = []
        for i in process_id_list:
            if i == 1132:
                y.append(1)
            else:
                y.append(0)
            indx_list = np.array((self.result['processID'] == i))
            r = []
            
            for j in range(len(indx_list)):
                if indx_list[j] == 1:
                    temp = callStack_result_temp[j]
                    r.extend(temp)
            r = list(set(r))
            callStack_result.append(r)
        
            

    
        print('----Phrase2------')
        for i in range(10):
            result = pd.read_csv('data/result2' + str(i) +'.csv')
            
            
            indexs = pd.notna(result['CallStack'])
            
            result = result[indexs]
            result = result[["processID", "CallStack"]]
            
            self.result = pd.concat([self.result, result], ignore_index=True)
            callStack = list(result['CallStack'])
            
            
            
            
            benigh_keyword = pd.read_csv('result/keyword_result_benign'+ str(i) + '.csv')
            reverse_keyword = pd.read_csv('result/keyword_result_reverse'+ str(i) +'.csv')
            
            benigh_keyword.sort_values(by=['importance'])
            reverse_keyword.sort_values(by=['importance'])
            
            reverse_size = int(reverse_keyword.shape[0]/10)
            benign_size = int(benigh_keyword.shape[0]/10)
            
            
            
            benigh_keyword_del = list(benigh_keyword[:benign_size]['keyword'])
            reverse_keyword_del = list(reverse_keyword[:reverse_size]['keyword'])
            
            
            
            callStack_result_temp = []
            for i in callStack:
                e = i[1:][:-1]
                temp = e.split(',')
                for i in range(len(temp)):
                    temp[i] = temp[i].strip("' ").strip(" '")
                    
                    if temp[i].lower() in benigh_keyword_del and temp[i].lower() in reverse_keyword_del:
                        temp[i] = ''
                    
                callStack_result_temp.append(temp)
            
            process_id_list = pd.unique(result['processID'])
            
            for i in process_id_list:
                if i == 10272:
                    for a in range(6):
                        y.append(1)
                else:
                    y.append(0)
                indx_list = np.array((result['processID'] == i))
                r = []
                if i != 10272:
                    for j in range(len(indx_list)):
                        if indx_list[j] == 1:
                            temp = callStack_result_temp[j]
                            r.extend(temp)
                    r = list(set(r))
                    callStack_result.append(r)
                else:
                    ind = np.nonzero(indx_list)[0]
                    a = np.random.permutation(ind.shape[0])
                    train_num = int(np.round(1/6 * len(a)))
                    for k in range(5):
                        r = []
                        ind_a = ind[k * train_num: (k+1) * train_num]
                        for o in ind_a:
                            temp = callStack_result_temp[o]
                            r.extend(temp)
                        r = list(set(r))
                        callStack_result.append(r)
                    r = []
                    k += 1
                    ind_a = ind[k * train_num: ]
                    for o in ind_a:
                        temp = callStack_result_temp[o]
                        r.extend(temp)
                    r = list(set(r))
                    callStack_result.append(r)


        print('---------------Phrase3-----------------')
        filenames=['1908','4696','7156','10568']
        for k in filenames:
            result = pd.read_csv('450 ETW Data/data/result'+k+'.csv')
            
            
            indexs = pd.notna(result['CallStack'])
            
            result = result[indexs]
            result = result[["processID", "CallStack"]]
            self.result = result
            callStack = list(result['CallStack'])
            
            
            
            benigh_keyword = pd.read_csv('result/keyword_result_benign'+k+'.csv')
            reverse_keyword = pd.read_csv('result/keyword_result_reverse'+k+'.csv')
            
            benigh_keyword.sort_values(by=['importance'])
            reverse_keyword.sort_values(by=['importance'])
            
            reverse_size = int(reverse_keyword.shape[0] /10)
            benign_size = int(benigh_keyword.shape[0] /10)
            
            
            
            benigh_keyword_del = list(benigh_keyword[:benign_size]['keyword'])
            reverse_keyword_del = list(reverse_keyword[:reverse_size]['keyword'])
            
            
            
            
            callStack_result_temp = []
            for i in callStack:
                e = i[1:][:-1]
                temp = e.split(',')
                for i in range(len(temp)):
                    temp[i] = temp[i].strip("' ").strip(" '")
                    
                    if temp[i].lower()  in benigh_keyword_del and temp[i].lower()  in reverse_keyword_del:
                        temp[i] = ''
                    
                callStack_result_temp.append(temp)

            process_id_list = pd.unique(result['processID'])
            
            for i in process_id_list:
                if i == int(k):
                    y.append(1)
                else:
                    y.append(0)
                indx_list = np.array((self.result['processID'] == i))
                r = []
                
                for j in range(len(indx_list)):
                    if indx_list[j] == 1:
                        temp = callStack_result_temp[j]
                        r.extend(temp)
                r = list(set(r))
                callStack_result.append(r)
                        
                    
                
        self.y = np.array(y)
        print(np.sum(y))
        n = len(max(callStack_result, key=len))
        callStack_result = [x + ['']*(n-len(x)) for x in callStack_result]
        self.callStack_result = callStack_result
        print('----Start Encodering---')
        self.one_hot_cleaning()
            
            
            
    def one_hot_cleaning(self):
        print(np.array(self.callStack_result).shape)
        api_list = np.unique(self.callStack_result)
        size = len(api_list)
        print(size)
        
        enc = LabelEncoder()
        enc.fit(api_list)
        
        self.X = []
        for i in self.callStack_result:
            temp = np.zeros(size)
            result = enc.transform(i)
            result = result[result!=0]
            for i in result:
                temp[i] = 1
            self.X.append(temp)


    
        

            
    def split_train_val_test(self):
        X = self.X
        y = self.y
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1)
        
        X_train, X_val, y_train, y_val  = train_test_split(X_train, y_train, test_size=0.25, random_state=1) 
        
    

        return X_train, X_val,X_test, y_train, y_val, y_test


