from optparse import OptionParser
import numpy as np
import time
from utils import *
from slow_lib import *
from ach_utils import *
from tc_utils import *
from ldpc_utils import *


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


### My (outer) code setting
###
parityLen = 8
assert (w+L*parityLen) % L ==0
J = int((w + L*parityLen)/L)                             # Length of each coded sub-block
messageLen = int(J - parityLen)
assert messageLen * L == w
M = 2**J                                            # Each coded sub-block is J-length binary, 
                                                        # to represent it in decimal, 
                                                        # it ranges in [0, M] = [0, 2**J].
windowSize = 2                                      # How many previous sections p(j) depends on
assert windowSize > 0
Pa = L*parityLen                                    # Total number of parity check bits, in this case Pa=w=128
listSize = K                                        # list size retained per section after AMP converges
parityInvolved = get_parity_involvement_matrix(L,windowSize,messageLen)    
                                                        # An L x L matrix.
                                                        # For each row i, the j-th entry = w/L(=8), iff, w(i) involves the construction of p(j). 
                                                        # E.g., parityInvolved[0] = [0,8,8,8,8,0,0,0,0,0,0,0,0,0,0,0]
whichGMatrix = get_G_matrices(parityInvolved)        # An L x L matrix. Only (i,j) s.t. parityInvolved[i][j]!=0 matters.
                                                        # For (i,j) of our interest, 
                                                        # whichGMatrix[i][j] returns a code (an index) for some specific G_{i,j} matrix.
                                                        # Where G_{i,j} matrix is the parity generating matrix needed to 
                                                        # calculate the contribution of w(i) while calculating p(j)
whichGMatrix_or = whichGMatrix.copy()



print("####### Start Rocking ######## K=" + str(K) +" and p_e= " + str(p_e))          # Simulation starts!!!!!
# Outer-code encoding. No need to change.
txBits = np.random.randint(low=2, size=(K, w))   


# LLC: Generate random binary messages for K active users. Hence txBits.shape is [K,w]
txBitsParitized_llc = slow_encode(txBits,K,L,J,Pa,w,messageLen,parityLen,parityInvolved,whichGMatrix) 
tx_symbols_llc = ach_binary_to_symbol(txBitsParitized_llc, L, K, J)


# * A-Channel with Erasure
seed = np.random.randint(0,10000)
rx_coded_symbols_llc, num_one_outage, one_outage_where, num_no_outage = a_plus_ch_with_erasure(tx_symbols_llc, L, K, J, p_e, seed=seed)
rx_coded_symbols_llc_or = rx_coded_symbols_llc.copy()
print(" Genie: How many no-outage ? " + str(num_no_outage))
print(" Genie: How many one-outage? " + str(num_one_outage))
print(" Genie: One-outage where: " + str(one_outage_where))


# Handling
print(" -Decode/correct now starts.")
tic = time.time()
rxBits_llc = llc_UACE_decoder(rx_coded_symbols_llc, L, J, messageLen, parityLen, listSize, parityInvolved, whichGMatrix, windowSize, APlus=True)
toc = time.time()
print(" | Time of correct " + str(toc-tic))

final_recovered_msgs = np.unique(rxBits_llc, axis=0)

# Check how many are true amongst those "corrected". No need to change.
_ = check(txBits, final_recovered_msgs, "Linked-loop Code", 2)


print(" -This simulation on A+Channel terminates.")