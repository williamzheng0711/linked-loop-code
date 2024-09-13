from optparse import OptionParser
import numpy as np
import time

from general_lib import *
from general_utils import *
from static_repo import *   # Contains all the static / constant stuff
from utils import *
from slow_lib import *
from abch_utils import *


p_e = 0
K = 50 
# channel_type = "B"
# SIC = True

txBits = np.random.randint(low=2, size=(K, B))        
seed = np.random.randint(1000)
print("The seed is " + str(seed))

### Run the simulation
simulation(16, 0.1, K, 3, "B", True, txBits, seed, 3)
simulation(16, 0.1, K, 2, "B", True, txBits, seed, 3)
simulation(15, 0.1, K, 2, "B", True, txBits, seed, 3)

simulation(16, 0.125, K, 3, "B", True, txBits, seed, 3)
simulation(16, 0.125, K, 2, "B", True, txBits, seed, 3)
simulation(15, 0.125, K, 2, "B", True, txBits, seed, 3)

simulation(16, 0.15, K, 3, "B", True, txBits, seed, 3)
simulation(16, 0.15, K, 2, "B", True, txBits, seed, 3)
simulation(15, 0.15, K, 2, "B", True, txBits, seed, 3)

print("********************Done*****************")