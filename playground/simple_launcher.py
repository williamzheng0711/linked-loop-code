from optparse import OptionParser
import numpy as np
import time

from general_lib import *
from general_utils import *
from static_repo import *   # Contains all the static / constant stuff
from utils import *
from slow_lib import *
from abch_utils import *


## Accept user inputs, specifying simulation arguments
parser = OptionParser()
parser.add_option("--args", type="string", dest="args", help="Arguments", default="")
parser.add_option("--Ka", type="int", dest="Ka", help="Number of active users", default=-1)
parser.add_option("--pe", type="float", dest="pe", help="Probability of being erased (mapped to emptyset)", default=-1)
parser.add_option("--L", type="int", dest="L", help="Number of sections", default=-1)
parser.add_option("--sic", type="int", dest="sic", help="Do SIC?", default=1)
parser.add_option("--M", type="int", dest="M", help="Window size?", default=-1)
parser.add_option("--ctype", type="string", dest="ctype", help="A or B?", default="None")
parser.add_option("--num_exp", type="int", dest="num_exp", help="Number of experiments", default=2)
parser.add_option("--toPrint", type="int", dest="toPrint", help="Whether to print", default=1)
(options, args) = parser.parse_args()


### Examine whether the user inputs are valid
K = options.Ka;                                 assert K > 0 
p_e = options.pe;                               assert p_e >= 0
L = options.L;                                  assert L in L_set
assert options.sic==0 or options.sic==1;        SIC = True if options.sic else False
M = options.M;                                  assert M in M_set # M = 2 or 3
channel_type = options.ctype;                   assert channel_type == "A" or channel_type == "B"
num_exp = options.num_exp;                      assert num_exp > 0
toPrint = True if options.toPrint else False


for _ in range(num_exp):
    ### Generate the iid random B-bit messages for each of the K users. Hence txBits.shape is [K,B]
    txBits = np.random.randint(low=2, size=(K, B))      
    random_seed = int(time.time() * 1000) % (2**32)

    ### Run the simulation
    simulation(L, p_e, K, M, channel_type, SIC, txBits, random_seed, toPrint=toPrint)