import numpy as np
from utils import *
import time
from fader_utils import *




# Parameter settings
w = 128                             # length of each user's uncoded message
L = 16                              # Number of sections/sub-blocks
parityLengthVector = np.array([0,7,7,8,8,8,8,8,8,8,8,8,8,9,9,16],dtype=int)
J=((w+np.sum(parityLengthVector))/L).astype(int) # Length of each coded sub-block
M=2**J                              # base case M = 2**16
messageLengthVector = np.subtract(J*np.ones(L, dtype = 'int'), parityLengthVector).astype(int)
Pa = np.sum(parityLengthVector)     # Total number of parity check bits
Ml = np.sum(messageLengthVector)    # Total number of information bits
K = 100                             # number of active users
N = int((30000 / 2**16)*M)          # number of channel uses (real d.o.f)
numAMPIter = 7                      # number of AMP iterations to perform
listSize = int(K + 5)               # list size retained per section after AMP converges
sigma_n = 1                         # AWGN noise standard deviation
SNR = 7                             # SNR (in dB) to play with
sigma_Rayleigh = 1                  # Rayleigh fading paremater
EbNo = 10**(SNR/10)                 # Eb/No
P = 2*w*EbNo/N                      # Power calculated
Phat = N*P/L                        # Power hat


# Outer code encoder and Rayleigh at users sides
parityDistribution, useWhichMatrix = generate_parity_distribution() 
txBits = np.random.randint(low=2, size=(K, w))                                      # Generate random messages for K active users
txBitsParitized = Tree_error_correct_encode(txBits, K,L,J,Pa,Ml,
                        messageLengthVector, parityLengthVector,parityDistribution) # add parities to txBits, get txBitsParitized
BETA = convert_bits_to_sparse_Rayleigh(txBitsParitized, L, J, K, sigma_Rayleigh)    # Rayleigh noises    


# Inner encode
Ab, Az = sparc_codebook(L, M, N)                        # Generate the binned SPARC codebook
innerOutput=Ab(BETA)    


# Channel Part.                               
x = np.sqrt(Phat)*innerOutput                           # x shape: (38400, 1) = (N, 1)
z = np.random.randn(N, 1) * sigma_n
y = (x + z).reshape(-1, 1)


# Inner code Decoder.
p0 = 1-(1-1/M)**K
decTempBETA = amp_prior_art_Rayleigh(y, sigma_n, P, L, M, numAMPIter, Ab, Az, p0, K, sigma_Rayleigh, False) 

## calculate and report genie statistics
analyze_genie_metrics(decTempBETA=decTempBETA, L=L, J=J, listSize=listSize, txBitsParitized=txBitsParitized, K=K)


# drop non significant values in beta. Ready for tree-code decoder.
decBetaSignificants, decBetaSignificantsPos = get_signi_values_and_positions(decTempBETA, L, J, listSize)


# Outer code (tree code) decoder 
tic = time.time()
rxBits, usedRootsIndex = Tree_decoder_fader(decBetaSignificants, decBetaSignificantsPos, L,J, w, parityLengthVector,messageLengthVector,listSize, parityDistribution)
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
print("correctly recovers " + str(thisIter) + " out of " +str(rxBits.shape[0]) )



#  Corrector
rxBits_corrected = Tree_corrector_fader(decBetaSignificants, decBetaSignificantsPos, L,J, w, parityLengthVector,messageLengthVector,listSize, parityDistribution, usedRootsIndex)
print("corrected shape: " + str( rxBits_corrected.shape))
print("txBits_remained shape is :" + str(txBits_remained.shape))

# Check how many are true amongst those "corrected"
corrected = 0
for i in range(txBits_remained.shape[0]):
    incre = 0
    incre = np.equal(txBits_remained[i,:],rxBits_corrected).all(axis=1).any()
    corrected += int(incre)
print("!!!!! CORRECTED " + str(corrected) + " out of " +str(rxBits_corrected.shape[0]) )

