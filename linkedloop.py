import numpy as np

class LinkedLoop:
    def __init__(self, arr, messageLen, lostPart):
        self.path = arr
        self.messageLen = messageLen
        self.lostPart = lostPart

    def get_path(self):
        return self.path
    
    def append_path(self, to_append):
        self.path = np.append(self.path, to_append)

    def get_lostPart(self):
        return self.lostPart
        
    # def write_lost(self, recovered_lost):
    #     assert len(recovered_lost) == self.messageLen
    #     self.lostPart = recovered_lost
    
    def whether_contains_na(self):
        flag = np.in1d(-1, self.path)
        return flag
    
    def get_messageLen(self):
        return self.messageLen