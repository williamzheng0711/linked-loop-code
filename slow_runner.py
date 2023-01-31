import numpy as np
import time
from utils import *
from slow_lib import *


# No. of active users & SNR settles the system
K = 10                                              # number of active users
SNR = 5                                             # SNR (in dB)


# Other parameter settings. No need to change at this moment.
w = 128                                             # Length of each user's uncoded message
L = 16                                              # Number of sections
parityLengthVector = int(w/L)*np.ones(L,dtype=int)  # As (outer code) rate is 1/2 at this moment
                                                        # and parities are distributed evenly into each section, 
                                                        # the number of info bits = 
                                                        # the number of parity bits for all sections.
J = ((w+np.sum(parityLengthVector))/L).astype(int)  # Length of each coded sub-block
M = 2**J                                            # Each coded sub-block is J-length binary, 
                                                        # to represent it in decimal, 
                                                        # it ranges in [0, M] = [0, 2**J].
messageLengthVector = np.subtract(J*np.ones(L,dtype='int'),parityLengthVector).astype(int)
Pa = np.sum(parityLengthVector)                     # Total number of parity check bits, in this case Pa=w=128
Ml = np.sum(messageLengthVector)                    # Total number of information bits (=w)
N = int((30000 / 2**16)*M)                          # number of channel uses (real d.o.f)
numAMPIter = 2                                      # number of AMP iterations desired
listSize = K + int(np.ceil(K/20))                   # list size retained per section after AMP converges
σ_n = 1                                             # AWGN noise standard deviation, hence set to 1. "n" stands for "normal"
EbNo = 10**(SNR/10)                                 # Eb/No
P = 2*w*EbNo/N                                      # Power calculated
Phat = N*P/L                                        # Power hat
σ_R = 1                                             # (standard) Rayleigh fading paremater. "R" stands for "Rayleigh"
                                                        # or σ in the formula given in 
                                                        # https://en.wikipedia.org/wiki/Rayleigh_distribution#Definition

parityInvolved = get_parity_involvement_matrix(L)    # An L x L matrix.
                                                        # For each row i, the j-th entry = w/L(=8), iff, w(i) involves the construction of p(j). 
                                                        # E.g., parityInvolved[0] = [0,8,8,8,8,0,0,0,0,0,0,0,0,0,0,0]

whichGMatrix = get_G_matrices(parityInvolved)        # An L x L matrix. Only (i,j) s.t. parityInvolved[i][j]!=0 matters.
                                                        # For (i,j) of our interest, 
                                                        # whichGMatrix[i][j] returns a code (an index) for some specific G_{i,j} matrix.
                                                        # Where G_{i,j} matrix is the parity generating matrix needed to 
                                                        # calculate the contribution of w(i) while calculating p(j)


print("####### Start Rocking #######")          # Simulation starts!!!!!


# Outer-code encoding. No need to change.
txBits = np.random.randint(low=2, size=(K, w))   
    # Generate random binary messages for K active users. Hence txBits.shape is [K,w]
txBitsParitized = slow_encode(txBits,K,L,J,Pa,Ml,messageLengthVector,parityLengthVector,parityInvolved,whichGMatrix) 
    # Add parities. txBitsParitized.size is (K,w+Pa)
β = convert_bits_to_sparse_Rayleigh(txBitsParitized,L,J,K,σ_R)     
    # Convert bits to sparse. Every user is multiplied by an iid Rayleigh distributed value


# *Inner-code encoding. No need to change.
Ab, Az = sparc_codebook(L, M, N)                    # Generate the SPARC codebook               
innerOutput = Ab(β)    


# *The channel Part. No need to change.                             
x = np.sqrt(Phat)*innerOutput                       # x is of size: (N, 1)
z = np.random.randn(N, 1) * σ_n                     # z is the Gaussian additive noise
y = (x + z).reshape(-1, 1)                         


# *Inner code decoder part. The Approximate message passing (AMP) that deals with Rayleigh. No need to change.
p0 = 1-(1-1/M)**K
print(" -AMP starts.")
estimated_β = amp_prior_art_Rayleigh(y,σ_n,P,L,M,numAMPIter,Ab,Az,p0,K,σ_R,False) 
print(" -AMP part is done.")


## *calculate and report genie statistics. No need to change. 
print(" -Genie part starts:")
analyze_genie_metrics(estimated_β,L,J,listSize,txBitsParitized,K)
print(" -Genie part is done.")


# *drop non-significant things in each section of estimated_β. Get ready for the decoder. No need to change.
sigValues, sigPos = get_sig_values_and_positions(estimated_β, L, J, listSize)
    # Note that in non-fading case, only the positions of significant values in estimated_β matter. 
    # However, when considering fading, the values themselves are also important because each individual has a fading number associated with him.
    # This can help us rule out (in a soft manner) some wrong paths when we come up with multiple valid paths.


# *Outer code decoder. PAINPOINT
print(" -Phase 1 (decoding) now starts.")
tic = time.time()
rxBits, usedRootsIndex = slow_decoder(sigValues,sigPos,L,J,w,parityLengthVector,messageLengthVector,listSize,parityInvolved,whichGMatrix)
toc = time.time()
print(" | Time of decode " + str(toc-tic))
if rxBits.shape[0] > K: 
    rxBits = rxBits[np.arange(K)]                    # As before, if we have >K paths, always choose the first K's.

print(usedRootsIndex)

# *Check how many is correct amongst the recover (recover means first phase). No need to change.
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
rxBits_corrected= slow_corrector(sigValues,sigPos,L,J,w,parityLengthVector,messageLengthVector,
                                listSize,parityInvolved,usedRootsIndex,whichGMatrix)
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