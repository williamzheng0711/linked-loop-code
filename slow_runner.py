import numpy as np
import time
from utils import *
from slow_lib import *


# Parameter settings
w = 128                                             # Length of each user's uncoded message
L = 16                                              # Number of sections
parityLengthVector=int(w/L)*np.ones(L,dtype=int)    # As (outer code) rate is 1/2 at this moment
                                                        # and parities are distributed evenly into each section, 
                                                        # the number of info bits = 
                                                        # the number of parity bits for all sections.
J=((w+np.sum(parityLengthVector))/L).astype(int)    # Length of each coded sub-block
M=2**J                                              # Each coded sub-block is J-length binary, 
                                                        # to represent it in decimal, 
                                                        # it ranges in [0, M] = [0, 2**J].
messageLengthVector=np.subtract(J*np.ones(L,dtype='int'),parityLengthVector).astype(int)
Pa = np.sum(parityLengthVector)                     # Total number of parity check bits, in this case Pa=w=128
Ml = np.sum(messageLengthVector)                    # Total number of information bits
K = 10                                              # number of active users
N = int((30000 / 2**16)*M)                          # number of channel uses (real d.o.f)
numAMPIter = 2                                      # number of AMP iterations desired
listSize = K + int(np.ceil(K/20))                   # list size retained per section after AMP converges
sigma_n = 1                                         # AWGN noise standard deviation, hence set to 1
SNR = 5                                             # SNR (in dB)
EbNo = 10**(SNR/10)                                 # Eb/No
P = 2*w*EbNo/N                                      # Power calculated
Phat = N*P/L                                        # Power hat
sigma_Rayleigh = 1                                  # (standard) Rayleigh fading paremater, 
                                                        # or σ in the formula given in https://en.wikipedia.org/wiki/Rayleigh_distribution#Definition
parityDistribution, useWhichMatrix = generate_parity_distribution_evenly(identity=False) 



print("----------Start Rocking----------")          # Simulation starts!!!!!


# Outer-code encoding
txBits = np.random.randint(low=2, size=(K, w))      # Generate random messages for K active users. txBits.size is (K,w)
txBitsParitized = Slow_encode(tx_message=txBits,    # Add parities. txBitsParitized.size is (K,w+Pa)
                                K=K, L=L, J=J, P=Pa, Ml=Ml, 
                                messageLengthVector=messageLengthVector, 
                                parityLengthVector=parityLengthVector, 
                                parityDistribution=parityDistribution, 
                                useWhichMatrix=useWhichMatrix) 

BETA = convert_bits_to_sparse_Rayleigh(txBitsParitized, L, J, K, sigma_Rayleigh)    # Rayleigh noises applied    


# Inner-code encoding
Ab, Az = sparc_codebook(L, M, N)                        # Generate the binned SPARC codebook
innerOutput=Ab(BETA)    


# Channel Part.                               
x = np.sqrt(Phat)*innerOutput                           # x shape: (38400, 1) = (N, 1)
z = np.random.randn(N, 1) * sigma_n
y = (x + z).reshape(-1, 1)


# Inner code Decoder. The Approximate message passing part.
p0 = 1-(1-1/M)**K
decTempBETA = amp_prior_art_Rayleigh(y, sigma_n, P, L, M, numAMPIter, Ab, Az, p0, K, sigma_Rayleigh, False) 


## calculate and report genie statistics
analyze_genie_metrics(decTempBETA=decTempBETA, L=L, J=J, listSize=listSize, txBitsParitized=txBitsParitized, K=K)


# drop non significant values in beta. Ready for tree-code decoder.
decBetaSignificants, decBetaSignificantsPos = get_signi_values_and_positions(decTempBETA, L, J, listSize)


# Outer code (tree code) decoder 
tic = time.time()
rxBits, usedRootsIndex = Slow_decoder_fader(L=L, J=J, B=w, 
                                            decBetaNoised=decBetaSignificants, decBetaPos=decBetaSignificantsPos, 
                                            parityLengthVector=parityLengthVector, messageLengthVector=messageLengthVector,
                                            listSize=listSize, parityDistribution=parityDistribution, useWhichMatrix=useWhichMatrix)
toc = time.time()
print("Time of new algo " + str(toc-tic))
if rxBits.shape[0] > K: 
    rxBits = rxBits[np.arange(K)]       # rxBits is what we decoded out in the first phase.



# Check how many is correct amongst the recover (recover means first phase)
thisIter = 0
txBits_remained = np.empty(shape=(0,0))
for i in range(txBits.shape[0]):
    incre = 0
    incre = np.equal(txBits[i,:],rxBits).all(axis=1).any()
    thisIter += int(incre)
    if (incre == False):
        txBits_remained = np.vstack( (txBits_remained, txBits[i,:]) ) if txBits_remained.size else  txBits[i,:]
print(" 1st phase: correctly recovers " + str(thisIter) + " out of " +str(rxBits.shape[0]) )



#  Corrector
rxBits_corrected = Slow_corrector_fader(decBetaNoised=decBetaSignificants, decBetaPos=decBetaSignificantsPos, 
                                        L=L, J=J, B=w, 
                                        parityLengthVector=parityLengthVector, messageLengthVector=messageLengthVector,
                                        listSize=listSize, parityDistribution=parityDistribution, usedRootsIndex= usedRootsIndex, 
                                        useWhichMatrix= useWhichMatrix)
print("corrected shape: " + str( rxBits_corrected.shape))
print("txBits_remained shape is :" + str(txBits_remained.shape))
print(rxBits_corrected)
if txBits_remained.shape[0] == w:
    txBits_remained = txBits_remained.reshape(1,-1)

# Check how many are true amongst those "corrected"
corrected = 0
if rxBits_corrected.size:
    for i in range(txBits_remained.shape[0]):
        incre = 0
        incre = np.equal(txBits_remained[i,:],rxBits_corrected).all(axis=1).any()
        corrected += int(incre)
    print("!!!!! CORRECTED " + str(corrected) + " out of " +str(rxBits_corrected.shape[0]) )
else: 
    print("Nothing was corrected")

# print(txBits_remained)
