from optparse import OptionParser
import numpy as np
import time

from general_lib import *
from general_utils import *
from static_repo import *
from utils import *
from slow_lib import *
from ach_utils import *
from tc_utils import *
from ldpc_utils import *

# Other parameter settings. No need to change at this moment.
w = 128                                             # Length of each user's uncoded message (total number of info bits)
J = 16                                              # Length of each coded sub-block

parser = OptionParser()
parser.add_option("--args", type="string", dest="args", help="Arguments", default="")
parser.add_option("--ka", type="int", dest="ka", help="Number of active users", default=-1)
parser.add_option("--pe", type="float", dest="pe", help="Probability of being erased (mapped to 0)", default=-1)
parser.add_option("--l", type="int", dest="l", help="Client's desired rate", default= 13)
(options, args) = parser.parse_args()

p_e = options.pe
assert p_e >= 0
K = options.ka                                      # number of active users
assert K > 0 
L = options.l 
assert L >=8 and L<=16 


### My (outer) code setting
###
messageLens, parityLens = get_allocation(L=L, J=J)
assert sum(messageLens) == w
M = 2**J                                            # Each coded sub-block is J-length binary, 
                                                        # to represent it in decimal, 
                                                        # it ranges in [0, M] = [0, 2**J].
windowSize = 2                                      # How many previous sections p(j) depends on
assert windowSize > 0
Pa = sum(parityLens)                                # Total number of parity check bits, in this case Pa=w=128
assert w + Pa == L*J

Gs, columns_index, sub_G_inversions = get_G_info(L=L)                                                    
Gijs, whichGMatrix = partitioning_Gs(L, Gs, parityLens, windowSize)

print("####### Start Rocking ######## K=" + str(K) +" and p_e= " + str(p_e))          # Simulation starts!!!!!
# Outer-code encoding. No need to change.
txBits = np.random.randint(low=2, size=(K, w))                              


# LLC: Generate random binary messages for K active users. Hence txBits.shape is [K,w]
txBitsParitized_llc = GLLC_encode(txBits,K,L,J,Pa,w,messageLens,parityLens, Gs, windowSize, Gijs)
tx_symbols_llc = GAch_binary_to_symbol(txBitsParitized_llc, L, K, J)


# * A-Channel with Erasure
seed = np.random.randint(0,10000)
rx_coded_symbols_plus, num_one_outage, one_outage_where, num_no_outage = APlus_ch_with_erasure(tx_symbols_llc, L, K, J, p_e, seed=seed)
rx_coded_symbols = remove_multiplicity(rx_coded_symbols_plus)

print(" Genie: How many no-outage ? " + str(num_no_outage))
print(" Genie: How many one-outage? " + str(num_one_outage))
print(" Genie: One-outage where: " + str(one_outage_where))

# Handling A+ 
print(" - A+ Channel: Decode/correct now starts.")
tic1 = time.time()
rxBits_llc = GLLC_UACE_decoder(rx_coded_symbols=rx_coded_symbols_plus, L=L, J=J, Gs=Gs, Gijs=Gijs, columns_index=columns_index, 
                               sub_G_inversions=sub_G_inversions, messageLens=messageLens, parityLens=parityLens, K=K,
                               windowSize=windowSize, whichGMatrix=whichGMatrix, APlus=True)
toc1 = time.time()
print(" | Time for A+ Channel " + str(toc1-tic1))
final_recovered_msgs = np.unique(rxBits_llc, axis=0)
# Check how many are true amongst those "corrected". No need to change.
_ = check(txBits, final_recovered_msgs, "Linked-loop Code", 2)
print(" -This simulation on A+Channel terminates.")




# Handling A Channel 
# print(" - A Channel: Decode/correct now starts.")
# tic2 = time.time()
# rxBits_llc = llc_UACE_decoder(rx_coded_symbols, L, J, messageLen, parityLen, listSize, parityInvolved, whichGMatrix_or, windowSize, APlus=False)
# toc2 = time.time()
# print(" | Time for A Channel " + str(toc2-tic2))
# final_recovered_msgs = np.unique(rxBits_llc, axis=0)
# # Check how many are true amongst those "corrected". No need to change.
# _ = check(txBits, final_recovered_msgs, "Linked-loop Code", 2)
# print(" -This simulation on A Channel terminates.")