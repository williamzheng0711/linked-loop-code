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
parser.add_option("--L", type="int", dest="L", help="Number of sections", default=-1)
parser.add_option("--sic", type="int", dest="sic", help="Do SIC?", default=1)
parser.add_option("--M", type="int", dest="M", help="Window size?", default=-1)
parser.add_option("--EbN0dB", type="float", dest="EbN0dB", help="SNR (dB)", default=1)

(options, args) = parser.parse_args()


phase = 3

### Examine whether the user inputs are valid
K = options.Ka;                                 assert K > 0 
L = options.L;                                  assert L in L_set
assert options.sic==0 or options.sic==1;        SIC = True if options.sic else False
M = options.M;                                  assert M in M_set # M = 2 or 3
EbNodB = options.EbN0dB;                        assert EbNodB >= 0

messageLens, parityLens = get_allocation(L=L);  N = 2**J # N denotes the length of a codeword, that is rate R = B / N
### Retrieve parity-generating matrices from matrix repository
Gis, columns_index, sub_G_invs = get_G_info(L, M, messageLens, parityLens, seed=42)
### Do partition on Gl's, making them into G_{l,l+1}, G_{l,l+2}, ... , G_{l,l+M}, these matrices with double subscripts are called Gijs
Gijs = partition_Gs(L, M, parityLens, Gis) 

###################################################################################################
### Simulation starts.
print("####### Start Rocking ######## K="+ str(K)+ " and L= "+ str(L) +" and M= " + str(M))                                

### Generate the iid random B-bit messages for each of the K users. Hence txBits.shape is [K,B]
txBits = np.random.randint(low=2, size=(K, B))        
seed = np.random.randint(1000)

### Encode all messages of K users. Hence tx_cdwds.shape is [K,N]
tx_cdwds = encode(txBits, K, L, N, M, messageLens, parityLens, Gijs)
### Convert binary coded-sub blocks to symbols
tx_symbols = binary_to_symbol(tx_cdwds, L, K)  ## This is a K times L matrix. 

### Convert symbols to sparse representation
β_0 = convert_bits_to_sparse_ours(tx_symbols,L,J,K)

## To do: replace this part with the actual Gaussian channel simulation
# Generate the binned SPARC codebook
n = 30000
p0 = 1-(1-1/(2**J))**K
maxSim=2 # number of simulations
msgDetected=0

# EbN0 in linear scale
EbNo = 10**(EbNodB/10)
P = 2*B*EbNo/n
σ_n = 1
# We assume equal power allocation for all the sections. Code has to be modified a little to accomodate different power allocations
Phat = n*P/L


Ab, Az = sparc_codebook(L, 2**J, n)

x = np.sqrt(Phat)*Ab(β_0)

# Generate random channel noise and thus also received signal y
z = np.random.randn(n, 1) * σ_n
y = (x + z).reshape(-1, 1)

# Run AMP decoding
T = 10
β = amp_prior_art(y, P, L, 2**J, T, Ab, Az,p0).reshape(-1)

# Convert decoded beta back to a message   
listSize = K
cs_decoded_tx_message = convert_sparse_to_bits(β,L,J,listSize)
# print(cs_decoded_tx_message.shape)

### Convert back to binary representation. (This is what in reality RX can get)
# grand_list = symbol_to_binary(K, L, cs_decoded_tx_message)
grand_list= cs_decoded_tx_message
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
all_decoded_txBits = np.unique(rxBits_p1, axis=0)
txBits_rmd_afterp1 = check_phase(txBits, all_decoded_txBits, "linked loop Code", "1")
if txBits_rmd_afterp1.shape[0] == B: # Only remains one message 
    txBits_rmd_afterp1 = txBits_rmd_afterp1.reshape(1,-1)
print(" -Phase 1 Done.\n")
###################################################################################################


###################################################################################################
### Decoding phase 2 (finding/recovering 1-outage codewords in the channel output) now starts.
print(" -- Decoding phase 2 now starts.")
tic = time.time()
rxBits_p21, grand_list= phase2plus_decoder(1, grand_list, L, Gis, columns_index, sub_G_invs, messageLens, parityLens, K, M, SIC=SIC)
toc = time.time()
print(" | Time of phase 2.1 " + str(toc-tic))
txBits_rmd_afterp21 = check_phase(txBits_rmd_afterp1, rxBits_p21, "Linked-loop Code", "2.1")

tic = time.time()
rxBits_p22, grand_list= phase2plus_decoder(1, grand_list, L, Gis, columns_index, sub_G_invs, messageLens, parityLens, K, M, SIC=SIC, pChosenRoots=[8])
toc = time.time()
print(" | Time of phase 2.2 " + str(toc-tic))
txBits_rmd_afterp22 = check_phase(txBits_rmd_afterp21, rxBits_p22, "Linked-loop Code", "2.2")

if rxBits_p21.size: 
    all_decoded_txBits = np.vstack((all_decoded_txBits, rxBits_p21)) if all_decoded_txBits.size else rxBits_p21
if rxBits_p22.size: 
    all_decoded_txBits = np.vstack((all_decoded_txBits, rxBits_p22)) if all_decoded_txBits.size else rxBits_p22
all_decoded_txBits = np.unique(all_decoded_txBits, axis=0)
_ = check_phase(txBits, all_decoded_txBits, "Linked-loop Code", "up-to-phase 2")
print(" -Phase 2 is done. \n")
#################################################################################################

if phase >=3:
###################################################################################################
### Decoding phase 3 (finding/recovering 2-outage codewords in the channel output) now starts.
    print(" -- Decoding phase 3 now starts.")
    tic = time.time()
    rxBits_p31, grand_list= phase2plus_decoder(2, grand_list, L, Gis, columns_index, sub_G_invs, messageLens, parityLens, K, M, SIC=SIC)
    toc = time.time()
    print(" | Time of phase 3.1 " + str(toc-tic))
    txBits_rmd_afterp31 = check_phase(txBits_rmd_afterp22, rxBits_p31, "Linked-loop Code", "3.1")

    tic = time.time()
    rxBits_p32, grand_list= phase2plus_decoder(2, grand_list, L, Gis, columns_index, sub_G_invs, messageLens, parityLens, K, M, SIC=SIC, pChosenRoots=[6])
    toc = time.time()
    print(" | Time of phase 3.2 " + str(toc-tic))
    txBits_rmd_afterp32 = check_phase(txBits_rmd_afterp31, rxBits_p32, "Linked-loop Code", "3.2")

    tic = time.time()
    rxBits_p33, grand_list= phase2plus_decoder(2, grand_list, L, Gis, columns_index, sub_G_invs, messageLens, parityLens, K, M, SIC=SIC, pChosenRoots=[6,10])
    toc = time.time()
    print(" | Time of phase 3.3 " + str(toc-tic))
    txBits_rmd_afterp33 = check_phase(txBits_rmd_afterp32, rxBits_p33, "Linked-loop Code", "3.3")

    all_decoded_txBits = np.vstack((all_decoded_txBits, rxBits_p31)) if rxBits_p31.size else  all_decoded_txBits
    all_decoded_txBits = np.vstack((all_decoded_txBits, rxBits_p32)) if rxBits_p32.size else  all_decoded_txBits
    all_decoded_txBits = np.vstack((all_decoded_txBits, rxBits_p33)) if rxBits_p33.size else  all_decoded_txBits
    all_decoded_txBits = np.unique(all_decoded_txBits, axis=0)
    _ = check_phase(txBits, all_decoded_txBits, "Linked-loop Code", "up-to-phase 3")
    print(" -Phase 3 is done, this simulation terminates.\n")
    #################################################################################################