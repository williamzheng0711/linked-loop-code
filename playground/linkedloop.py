import numpy as np

class GLinkedLoop:
    def __init__(self, list, messageLens, listLostSects=None, dictLostInfos=None):
        self.path = list
        self.messageLens = messageLens
        self.listLostSects = listLostSects if listLostSects is not None else []
        self.dictLostInfos = dictLostInfos if dictLostInfos is not None else {}
    def get_path(self):
        return self.path

    # This returns the dictionary containing recovered lost bits for respective sections
    def get_dictLostInfos(self):
        return self.dictLostInfos
    
    # Returns the list containing lost sections
    def get_listLostSects(self):
        return self.listLostSects
    
    def num_na_in_path(self):
        return self.path.count(-1)
    
    def get_messageLens(self):
        return self.messageLens
    
    def known_in_path(self, index):
        if index >= len(self.path):
            return False
        else:
            return True
        
    def all_known(self):
        flag = True
        for l in range(len(self.path)):
            if self.path[l] == -1:
                if l not in self.dictLostInfos:
                    flag = False
                    return flag
        return flag
        
