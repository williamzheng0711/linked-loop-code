from optparse import OptionParser
import numpy as np
import time

from general_lib import *
from general_utils import *
from static_repo import *   # Contains all the static / constant stuff
from utils import *
from slow_lib import *
from ach_utils import *
from tc_utils import *
from ldpc_utils import *

### Accept user inputs, specifying simulation arguments
parser = OptionParser()
parser.add_option("--args", type="string", dest="args", help="Arguments", default="")
parser.add_option("--Ka", type="int", dest="Ka", help="Number of active users", default=-1)
parser.add_option("--pe", type="float", dest="pe", help="Probability of being erased (mapped to emptyset)", default=-1)
parser.add_option("--L", type="int", dest="L", help="Number of sections", default=-1)
parser.add_option("--sic", type="int", dest="sic", help="Do SIC?", default=1)
parser.add_option("--M", type="int", dest="M", help="Window size?", default=-1)
parser.add_option("--ctype", type="string", dest="ctype", help="A or B?", default="None")
(options, args) = parser.parse_args()

### Examine whether the user inputs are valid
K = options.Ka;                                 assert K > 0 
p_e = options.pe;                               assert p_e >= 0
L = options.L;                                  assert L in L_set
assert options.sic==0 or options.sic==1;        SIC = True if options.sic else False
M = options.M;                                  assert M in M_set # M = 2 or 3
channel_type = options.ctype;                   assert channel_type == "A" or channel_type == "B"

### Extract pre-determined info-parity pattern
messageLens, parityLens = get_allocation(L=L);  N = 2**J # N denotes the length of a codeword, that is rate R = B / N
### Retrieve parity-generating matrices from matrix repository
Gs, columns_index, sub_G_inversions = get_G_info(L, M, messageLens, parityLens)
Gijs, whichGMatrix = partitioning_Gs(L, Gs, parityLens, M) 

print("####### Start Rocking ######## K=" + str(K) +" and p_e= " + str(p_e) + " and L= " + str(L) +" and windowSize= " + str(windowSize))          # Simulation starts!!!!!
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
txBits_remained_llc = check_phase(txBits, rxBits_llc, "Linked-loop Code", "1")
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


all_decoded_txBits = np.vstack( ( rxBits_llc,rxBits_corrected_llc   ) )
all_decoded_txBits = np.unique(all_decoded_txBits, axis=0)
_ = check_phase(txBits, all_decoded_txBits, "Linked-loop Code", "all")

print(" -Phase 2 is done, this simulation terminates.")
############################