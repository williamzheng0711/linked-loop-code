import numpy as np

class LinkedLoop:
    def __init__(self, list, messageLen, lostPart=None):
        self.path = list
        self.messageLen = messageLen
        self.lostPart = lostPart if lostPart is not None else -1*np.ones((messageLen),dtype=int)

    def get_path(self):
        return self.path

    def get_lostPart(self):
        return self.lostPart
    
    def whether_contains_na(self):
        return -1 in self.path
    
    def get_messageLen(self):
        return self.messageLen