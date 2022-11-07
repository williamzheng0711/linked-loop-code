import numpy as np

oldPaths = np.array([[16,23],[16,77]])

validPos = np.array([1,4,3,90,21])
newPaths = []

for i in np.arange(oldPaths.shape[0]):
    aa = np.tile(oldPaths[i], (len(validPos),1))
    newPaths.append( np.hstack( ( aa, validPos.reshape(-1,1)) ))

newPaths = np.array(newPaths).reshape()

print(newPaths)