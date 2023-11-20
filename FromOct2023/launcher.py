from optparse import OptionParser
import numpy as np
import time

from general_lib import *
from general_utils import *
from static_repo import *   # Contains all the static / constant stuff
from utils import *
from slow_lib import *
from abch_utils import *


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
Gis, columns_index, sub_G_invs = get_G_info(L, M, messageLens, parityLens)
### Do partition on Gl's, making them into G_{l,l+1}, G_{l,l+2}, ... , G_{l,l+M}, these matrices with double subscripts are called Gijs
Gijs = partition_Gs(L, M, parityLens, Gis) 




###################################################################################################
### Simulation starts.
print("####### Start Rocking ######## K="+ str(K)+ " and p_e= "+ str(p_e)+ " and L= "+ str(L) +" and M= " + str(M))          
### Generate the iid random B-bit messages for each of the K users. Hence txBits.shape is [K,B]
txBits = np.random.randint(low=2, size=(K, B))                              

### Encode all messages of K users. Hence tx_cdwds.shape is [K,N]
tx_cdwds = encode(txBits, K, L, N, M, messageLens, parityLens, Gijs)
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
print(" Genie: How many 0-outage? " + str(n0))
print(" Genie: How many 1-outage? " + str(n1))
print(" Genie: How many 2-outage? " + str(n2))
print(" Genie: 1-outage positions: " + str(one_outage_where))
### Convert back to binary representation. (This is what in reality RX can get)
grand_list = symbol_to_binary(K, L, rx_symbols)
###################################################################################################



###################################################################################################
### Decoding phase 1 (simply finding & stitching 0-outage codewords in the channel output) now starts.
print(" -- Decoding phase 1 now starts.")
tic = time.time()
rxBits_p1, grand_list = phase1_decoder(grand_list, L, Gijs, messageLens, parityLens, K, M, SIC=SIC)
toc = time.time()
print(" | Time of phase 1 (LLC): " + str(toc-tic))

### If we have >K decoded messages, only choose the first K.
if rxBits_p1.shape[0] > K: 
    rxBits_p1 = rxBits_p1[np.arange(K)]                    

### Check how many are correct amongst the recover (recover means first phase). No need to change.
txBits_rmd_afterp1 = check_phase(txBits, rxBits_p1, "linked loop Code", "1")
if txBits_rmd_afterp1.shape[0] == B: # Only remains one message 
    txBits_rmd_afterp1 = txBits_rmd_afterp1.reshape(1,-1)
print(" -Phase 1 Done.\n")
###################################################################################################




###################################################################################################
### Decoding phase 2 (finding/recovering 1-outage codewords in the channel output) now starts.
print(" -- Decoding phase 2 now starts.")
tic = time.time()
rxBits_p21, grand_list= phase2_decoder(grand_list, L, Gis, Gijs, columns_index, sub_G_invs, messageLens, parityLens, K, M, SIC=SIC, erasure_slot=None)
toc = time.time()
print(" | Time of phase 2.1 " + str(toc-tic))
txBits_rmd_afterp21 = check_phase(txBits_rmd_afterp1, rxBits_p21, "Linked-loop Code", "2.1")

tic = time.time()
rxBits_p22, grand_list= phase2_decoder(grand_list, L, Gis, Gijs, columns_index, sub_G_invs, messageLens, parityLens, K, M, SIC=SIC, pChosenRoot=7, erasure_slot=0)
toc = time.time()
print(" | Time of phase 2.2 " + str(toc-tic))
txBits_rmd_afterp22 = check_phase(txBits_rmd_afterp21, rxBits_p22, "Linked-loop Code", "2.2")
print(" | \n")

all_decoded_txBits = np.vstack((rxBits_p1, rxBits_p21, rxBits_p22)) if rxBits_p22.size else np.vstack((rxBits_p1, rxBits_p21))
all_decoded_txBits = np.unique(all_decoded_txBits, axis=0)
_ = check_phase(txBits, all_decoded_txBits, "Linked-loop Code", "up-to-phase 2")

print(" -Phase 2 is done, this simulation terminates.")
#################################################################################################





###################################################################################################
### Decoding phase 2plus (finding/recovering 2-outage codewords in the channel output) now starts.
print(" -- Decoding phase 2+ now starts.")


