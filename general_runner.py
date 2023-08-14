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
parser.add_option("--l", type="int", dest="l", help="Client's desired rate", default=-1)
parser.add_option("--sic", type="int", dest="sic", help="Do SIC?", default=-1)
parser.add_option("--ws", type="int", dest="ws", help="Window size?", default=-1)
parser.add_option("--ctype", type="string", dest="ctype", help="A or B?", default="None")
(options, args) = parser.parse_args()

p_e = options.pe
assert p_e >= 0
windowSize = options.ws
assert windowSize > 1
K = options.ka                                      # number of active users
assert K > 0 
L = options.l 
assert L >=8 and L<=16 
channel_type = options.ctype
assert channel_type == "A" or channel_type == "B"

assert options.sic == 0 or options.sic == 1
SIC = True if options.sic else False


### My (outer) code setting
###
messageLens, parityLens = get_allocation(L=L, J=J)
assert sum(messageLens) == w
M = 2**J                                            # Each coded sub-block is J-length binary, 
                                                        # to represent it in decimal, 
                                                        # it ranges in [0, M] = [0, 2**J].
assert windowSize > 0
Pa = sum(parityLens)                                # Total number of parity check bits, in this case Pa=w=128
assert w + Pa == L*J

Gs, columns_index, sub_G_inversions = get_G_info(L=L, windowSize=windowSize)                                                    
Gijs, whichGMatrix = partitioning_Gs(L, Gs, parityLens, windowSize)

print("####### Start Rocking ######## K=" + str(K) +" and p_e= " + str(p_e) + " and L= " + str(L))          # Simulation starts!!!!!
# Outer-code encoding. No need to change.
txBits = np.random.randint(low=2, size=(K, w))                              


# LLC: Generate random binary messages for K active users. Hence txBits.shape is [K,w]
txBitsParitized_llc = GLLC_encode(txBits,K,L,J,Pa,w,messageLens,parityLens, Gs, windowSize, Gijs)
tx_symbols_llc = GAch_binary_to_symbol(txBitsParitized_llc, L, K, J)


# * A-Channel with Erasure
seed = np.random.randint(0,10000)
rx_coded_symbols, num_one_outage, one_outage_where, num_no_outage = APlus_ch_with_erasure(tx_symbols_llc, L, K, J, p_e, seed=seed)
if channel_type == "A":
    rx_coded_symbols = remove_multiplicity(rx_coded_symbols)

print(" Genie: How many no-outage ? " + str(num_no_outage))
print(" Genie: How many one-outage? " + str(num_one_outage))
print(" Genie: One-outage where: " + str(one_outage_where))


print(" -Phase 1 (decoding) now starts.")
tic = time.time()
rxBits_llc, cs_decoded_tx_message, num_erase = GLLC_UACE_decoder(rx_coded_symbols=rx_coded_symbols, L=L, J=J, 
                                                                 Gijs=Gijs, messageLens=messageLens, parityLens=parityLens, 
                                                                 K=K, windowSize=windowSize, whichGMatrix=whichGMatrix, SIC=SIC)
toc = time.time()
print(" | Time of GLLC decode " + str(toc-tic))
if rxBits_llc.shape[0] > K: 
    rxBits_llc = rxBits_llc[np.arange(K)]                    # As before, if we have >K paths, always choose the first K's.

# Check how many are correct amongst the recover (recover means first phase). No need to change.
txBits_remained_llc = check_phase_1(txBits, rxBits_llc, "Linked-loop Code")
print(" -Phase 1 Done.")


# *Corrector. PAINPOINT
print(" -Phase 2 (correction) now starts.")
tic = time.time()
rxBits_corrected_llc= GLLC_UACE_corrector(cs_decoded_tx_message=cs_decoded_tx_message, L=L, J=J, Gs=Gs, Gijs=Gijs, columns_index=columns_index, 
                                        sub_G_inversions=sub_G_inversions, messageLens=messageLens, parityLens=parityLens, K=K,
                                        windowSize=windowSize, whichGMatrix=whichGMatrix, num_erase=num_erase, SIC=SIC)
toc = time.time()
print(" | Time of correct " + str(toc-tic))
if txBits_remained_llc.shape[0] == w:
    txBits_remained_llc = txBits_remained_llc.reshape(1,-1)

# Check how many are true amongst those "corrected". No need to change.
corrected = 0
if rxBits_corrected_llc.size:
    for i in range(txBits_remained_llc.shape[0]):
        incre = 0
        incre = np.equal(txBits_remained_llc[i,:],rxBits_corrected_llc).all(axis=1).any()
        corrected += int(incre)
    print(" | In phase 2, Linked-loop code corrected " + str(corrected) + " true (one-outage) message out of " +str(rxBits_corrected_llc.shape[0]) )
else: 
    print(" | Nothing was corrected")

print(" -Phase 2 is done, this simulation terminates.")
print("   ")