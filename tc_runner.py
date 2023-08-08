from optparse import OptionParser
import numpy as np
import time
from utils import *
from slow_lib import *
from ach_utils import *
from tc_utils import *
from ldpc_utils import *


## This is the tree code simulator. 
## Rate is fixed as 1/2
## A for A-Channel, B for B-Channel (aka. A plus)


parser = OptionParser()
parser.add_option("--args", type="string", dest="args", help="Arguments", default="")
parser.add_option("--ka", type="int", dest="ka", help="Number of active users", default=-1)
parser.add_option("--pe", type="float", dest="pe", help="Probability of being erased (mapped to 0)", default=-1)
(options, args) = parser.parse_args()

p_e = options.pe
assert p_e >= 0
K = options.ka                                      # number of active users
assert K > 0 

# Other parameter settings. No need to change at this moment.
w = 128                                             # Length of each user's uncoded message (total number of info bits)
L = 16                                              # Number of sections
assert w % L ==0

### Tree (outer) code setting 
### This is obtained from the Opt_prob.m file


parityLengthVector = np.array([0, 6, 8,8,8,8,8,8,8,8,8,8,8,8, 10, 16],dtype=int) # Parity bits distribution

J=((w+np.sum(parityLengthVector))/L).astype(int) # Length of each coded sub-block
M=2**J # Length of each section


txBits = np.random.randint(low=2, size=(K, w)) 

rx_coded_symbols, num_one_outage, one_outage_where, num_no_outage = APlus_ch_with_erasure(tx_symbols_llc, L, K, J, p_e, seed=seed)

