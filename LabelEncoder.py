import pickle

class LabelEncoder:
    '''
    Utility function to encode labels to keys and vice versa
    '''
    def __init__(self,label_list):
        self.label_list = list(set(label_list))
        self.map_key = {index:self.label_list[index] for index in range(len(self.label_list))}
        self.unmap_key = {self.map_key[idx]:idx for idx in range(len(self.label_list))}
        
    def get_label(self,indexs):
        result = []
        for i in indexs:
            result.append(self.map_key[i])
        return result

    def get_index(self,labels):
        result = []
        for i in labels:
            result.append(self.unmap_key[i])
        return result
     
    def save(self,path = 'LabelEncoder'):
        pickle.dump(self,open(path,'wb')) 
    
    @staticmethod
    def load_from_pickle(path):
        return pickle.load(open(path,'rb'))