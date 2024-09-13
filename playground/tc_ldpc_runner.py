from optparse import OptionParser
import numpy as np
import time
from utils import *
from slow_lib import *
from abch_utils import *
from tc_utils import *
from ldpc_utils import *
from general_lib import *


parser = OptionParser()
parser.add_option("--args", type="string", dest="args", help="Arguments", default="")
parser.add_option("--ka", type="int", dest="ka", help="Number of active users", default=-1)
parser.add_option("--pe", type="float", dest="pe", help="Probability of being erased (mapped to 0)", default=-1)
parser.add_option("--ctype", type="string", dest="ctype", help="A or B?", default="None")
parser.add_option("--sic", type="int", dest="sic", help="Do SIC?", default=-1)


(options, args) = parser.parse_args()
assert options.sic == 0 or options.sic == 1
SIC = True if options.sic else False
channel_type = options.ctype
assert channel_type == "A" or channel_type == "B"
p_e = options.pe
assert p_e >= 0
K = options.ka                                      # number of active users
assert K > 0
# Other parameter settings. No need to change at this moment.
w = 128                                             # Length of each user's uncoded message (total number of info bits)
L = 16                                              # Number of sections
assert w % L ==0


### Tree (outer) code setting
###
### This is obtained from the Opt_prob.m file
parityLengthVector = np.array([0, 6, 8,8,8,8,8,8,8,8,8,8,8,8, 10, 16],dtype=int) # Parity bits distribution
J=((w+np.sum(parityLengthVector))/L).astype(int) # Length of each coded sub-block
M=2**J # Length of each section
messageLengthVector = np.subtract(J*np.ones(L, dtype = 'int'), parityLengthVector).astype(int)
Pa = np.sum(parityLengthVector) # Total number of parity check bits
Ml = np.sum(messageLengthVector) # Total number of information bits
G = generate_parity_matrix(L,messageLengthVector,parityLengthVector)


print("####### TC & LDPC runner: Start Rocking ########  K=" + str(K) +" and p_e= " + str(p_e))          # Simulation starts!!!!!
# Outer-code encoding. No need to change.
txBits = np.random.randint(low=2, size=(K, w))

# Tree Code: Encode and from binary to symbols
txBitsParitized_tc = Tree_encode(txBits,K,G,L,J,Pa,Ml,messageLengthVector,parityLengthVector)
tx_symbols_tc = GAch_binary_to_symbol(txBitsParitized_tc, L, K, J)

# # LDPC: Get encoded symbols
# outer_code = FGG.Triadic8(16)
# tx_symbols_ldpc, user_codewords = LDPC_encode_to_symbol(txBits, L, K, J, outer_code)

# * A-Channel with Deletion
seed = np.random.randint(0,10000)
rx_coded_symbols_tc  = ach_with_erasure(tx_symbols_tc,  L, K, J, p_e, seed=seed)
rx_coded_symbols_tc, _, _, _ = APlus_ch_with_erasure(tx_symbols_tc, L, K, J, p_e, seed=seed)
if channel_type == "A":
    rx_coded_symbols_tc = remove_multiplicity(rx_coded_symbols_tc)

# rx_coded_symbols_ldpc= ach_with_erasure(tx_symbols_ldpc,L, K, J, p_e, seed=seed)
# rx_coded_symbols_ldpc, _, _, _ = APlus_ch_with_erasure(tx_symbols_ldpc, L, K, J, p_e, seed=seed)
# if channel_type == "A":
#     rx_coded_symbols_ldpc = remove_multiplicity(rx_coded_symbols_ldpc)


# *Outer code decoder.
## Tree Code
tic = time.time()
cs_decoded_tc = Tree_symbols_to_bits(K, L, J, rx_coded_symbols_tc)
rxBits_tc = Tree_decoder(cs_decoded_tc,G,L,J,w,parityLengthVector,messageLengthVector,K, SIC=SIC)
toc = time.time()
print(" | Time of Tree Code decode " + str(toc-tic))
if rxBits_tc.shape[0] > K:
    rxBits_tc = rxBits_tc[np.arange(K)]


# ## LDPC
# tic = time.time()
# unioned_cdwds_ldpc = LDPC_symbols_to_bits(L, J, rx_coded_symbols_ldpc, K, channel_type)
# rx_user_codewords = outer_code.decoder(unioned_cdwds_ldpc, K)
# rx_user_codewords = np.array(rx_user_codewords)
# toc = time.time()
# print(" | Time of LDPC Code decode " + str(toc-tic))

## Tree code
_                   = check_phase_1(txBits, rxBits_tc, "Tree Code")

# ## LDPC code
# LDPC_num_matches = FGG.numbermatches(user_codewords, rx_user_codewords, K)
# print(f' | In phase 1, LDPC decodes {LDPC_num_matches}/{len(rx_user_codewords)} codewords. ')
# print(" -Phase 1 Done.")

print(" -This simulation terminates.")