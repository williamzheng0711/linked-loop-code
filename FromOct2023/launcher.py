from optparse import OptionParser
import numpy as np
import time

from general_lib import *
from general_utils import *
from static_repo import *   # Contains all the static / constant stuff
from utils import *
from slow_lib import *
from abch_utils import *
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
Gis, columns_index, sub_G_inversions = get_G_info(L, M, messageLens, parityLens)
### Do partition on Gl's, making them into G_{l,l+1}, G_{l,l+2}, ... , G_{l,l+M}, these matrices with double subscripts are called Gijs
Gijs, Gij_cipher = partition_Gs(L, M, parityLens, Gis) 

### Simulation starts.
print("####### Start Rocking ######## K="+ str(K)+ " and p_e= "+ str(p_e)+ " and L= "+ str(L) +" and M= " + str(M))          
### Generate the iid random B-bit messages for each of the K users. Hence txBits.shape is [K,B]
txBits = np.random.randint(low=2, size=(K, B))                              

### Encode all messages of K users. Hence tx_cdwds.shape is [K,N]
tx_cdwds = GLLC_encode(txBits, K, L, N, M, messageLens, parityLens, Gijs)
### Convert binary coded-sub blocks to symbols
tx_symbols = binary_to_symbol(tx_cdwds, L, K)

### B-Channel with Erasure
seed = np.random.randint(0,10000) # randomness for erasure pattern
rx_symbols, one_outage_where, n0, n1, n2 = bch_with_erasure(tx_symbols, L, K, p_e, seed=seed)
if channel_type == "A":
    # A-channel is obtained by removing multiplicities from B-channel
    # We call "rx_symbols" or its equivalence as "the grand list"
    rx_symbols = remove_multiplicity(rx_symbols)

### Generate genie reports
print(" Genie: How many 0-outage ? " + str(n0))
print(" Genie: How many 1-outage? " + str(n1))
print(" Genie: 1-outage positions: " + str(one_outage_where))

### Decoding phase 1 (simply finding & stitching 0-outage codewords in the channel output) now starts.
print(" -- Decoding phase 1 now starts.")
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