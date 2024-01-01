from optparse import OptionParser
import numpy as np
import time

from general_lib import *
from general_utils import *
from static_repo import *   # Contains all the static / constant stuff
from utils import *
from slow_lib import *
from abch_utils import *


p_e = 0.075
K = 50 
channel_type = "B"
SIC = True

txBits = np.random.randint(low=2, size=(K, B))        
seed = np.random.randint(1000)

### Run the simulation
simulation(16, p_e, K, 2, channel_type, SIC, txBits, seed)
simulation(16, p_e, K, 3, channel_type, SIC, txBits, seed)
simulation(15, p_e, K, 2, channel_type, SIC, txBits, seed)
simulation(15, p_e, K, 3, channel_type, SIC, txBits, seed)

print("********************Done*****************")