from optparse import OptionParser
import numpy as np
import time
from utils import *
from slow_lib import *
from ach_utils import *



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

print("####### Start Rocking ######## K=" + str(K) +" and p_e= " + str(p_e))          # Simulation starts!!!!!
# Outer-code encoding. No need to change.
txBits = np.random.randint(low=2, size=(K, w))   
# Generate random binary messages for K active users. Hence txBits.shape is [K,w]
txBitsParitized = slow_encode(txBits,K,L,J,Pa,w,messageLen,parityLen,parityInvolved,whichGMatrix) 
# Add parities. txBitsParitized.size is (K,w+Pa)

tx_symbols = ach_binary_to_symbol(txBitsParitized, L, K, J)
# txBits_GF.shape should be (K,L), each slot should be a number of range [0,2**J)


# * A-Channel with Deletion
rx_coded_symbols = ach_with_deletion(tx_symbols, L, K, J, p_e)
# rx_coded_symbols 裡面有 -1的話說明是deletion了

# *Outer code decoder. PAINPOINT
print(" -Phase 1 (decoding) now starts.")
tic = time.time()
rxBits, usedRootsIndex, listSizeOrder = slow_decoder(np.ones((listSize,L),dtype=int), rx_coded_symbols, L, J, parityLen, messageLen, listSize, parityInvolved, whichGMatrix, windowSize)
# print(usedRootsIndex)

toc = time.time()
print(" | Time of decode " + str(toc-tic))
if rxBits.shape[0] > K: 
    rxBits = rxBits[np.arange(K)]                    # As before, if we have >K paths, always choose the first K's.

# Check how many are correct amongst the recover (recover means first phase). No need to change.
thisIter = 0
txBits_remained = np.empty(shape=(0,0))
for i in range(txBits.shape[0]):
    incre = 0
    incre = np.equal(txBits[i,:],rxBits).all(axis=1).any()
    thisIter += int(incre)
    if (incre == False):
        txBits_remained = np.vstack( (txBits_remained, txBits[i,:]) ) if txBits_remained.size else  txBits[i,:]
print(" | In phase 1, we decodes " + str(thisIter) + " true message out of " +str(rxBits.shape[0]))
print(" -Phase 1 is done.")



# *Corrector. PAINPOINT
print(" -Phase 2 (correction) now starts.")
tic = time.time()
rxBits_corrected= slow_corrector(np.ones((listSize,L),dtype=int),rx_coded_symbols,L,J,messageLen,parityLen,listSize,parityInvolved,usedRootsIndex,whichGMatrix,windowSize,listSizeOrder)
toc = time.time()
print(" | Time of correct " + str(toc-tic))
print(" | corrected shape: " + str( rxBits_corrected.shape))
print(" | txBits_remained shape is :" + str(txBits_remained.shape))

if txBits_remained.shape[0] == w:
    txBits_remained = txBits_remained.reshape(1,-1)



# Check how many are true amongst those "corrected". No need to change.
corrected = 0
if rxBits_corrected.size:
    for i in range(txBits_remained.shape[0]):
        incre = 0
        incre = np.equal(txBits_remained[i,:],rxBits_corrected).all(axis=1).any()
        corrected += int(incre)
    print(" | In phase 2, we corrected " + str(corrected) + " true (one-outage) message out of " +str(rxBits_corrected.shape[0]) )
else: 
    print(" | Nothing was corrected")

print(" -Phase 2 is done, this simulation terminates.")