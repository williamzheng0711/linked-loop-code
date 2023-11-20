import numpy as np

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
    
    def num_na_in_path(self):
        return self.path.count(-1)
    
    def get_messageLens(self):
        return self.messageLens
    
    def known_in_path(self, index):
        if index >= len(self.path):
            return False
        else:
            return True