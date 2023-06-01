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



class GLinkedLoop:
    def __init__(self, list, messageLens, lostPart=None, lostSection=None):
        self.path = list
        self.messageLens = messageLens
        self.lostSection = lostSection if lostSection is not None else -1
        self.lostPart = lostPart if lostPart is not None else np.empty((0),dtype=int)

    def get_path(self):
        return self.path

    def get_lostPart(self):
        return self.lostPart
    
    def get_lostSection(self):
        return self.lostSection
    
    def whether_contains_na(self):
        return -1 in self.path
    
    def get_messageLens(self):
        return self.messageLens