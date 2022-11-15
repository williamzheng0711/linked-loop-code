from multiprocessing.heap import Arena
import time
import numpy as np
from utils import *
from fader_utils import *

w = 128     # is called B in uninformative         # length of each user's uncoded message
L = 16      # Number of sections/sub-blocks

parityLengthVector = np.array([8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8],dtype=int)

parityDistribution = generate_parity_distribution_evenly()
# print(parityDistribution)



J=((w+np.sum(parityLengthVector))/L).astype(int) # Length of each coded sub-block
M=2**J # base case M = 2**16

messageLengthVector = np.subtract(J*np.ones(L, dtype = 'int'), parityLengthVector).astype(int)

Pa = np.sum(parityLengthVector) # Total number of parity check bits
Ml = np.sum(messageLengthVector) # Total number of information bits


K = 100                                             # number of active users
# N = 38400                                           # number of channel uses (real d.o.f)
N = int((30000 / 2**16)*M)
numAMPIter = 10                                     # number of AMP iterations to perform
listSize = int(K + 5)                                     # list size retained per section after AMP converges
sigma_n = 1                                         # AWGN noise standard deviation


SNR = 7
sigma_Rayleigh = 1

EbNo = 10**(SNR/10)
P = 2*w*EbNo/N

Phat = N*P/L

# Generate random messages for K active users
txBits = np.random.randint(low=2, size=(K, w))

# Up till now, I hard code the message -> outer code procedure.
txBitsParitized = Tree_error_correct_encode_tb(txBits, K,L,J,Pa,Ml,
                        messageLengthVector, parityLengthVector,parityDistribution)

# Convert bits to sparse representation
# Note that BETA has shape e.g. (L*2**J,1)=(1048576, 1)
BETA = convert_bits_to_sparse_Rayleigh(txBitsParitized, L, J, K, sigma_Rayleigh)
print("non zero: " + str(sum([1 if i!=0 else 0 for i in BETA])))
print("beta with fading: sum is: " + str(sum(BETA)) + " len: " + str(len(BETA)) )

# Generate the binned SPARC codebook
Ab, Az = sparc_codebook(L, M, N)

innerOutput=Ab(BETA)
x = np.sqrt(Phat)*innerOutput  # x shape: (38400, 1) = (N, 1)
# Generate random channel noise and thus also received signal y
z = np.random.randn(N, 1) * sigma_n
y = (x + z).reshape(-1, 1)

p0 = 1-(1-1/M)**K

# AMP does not need to change.
decTempBETA = amp_prior_art_Rayleigh(y, sigma_n, P, L, M, numAMPIter, Ab, Az, p0, K, sigma_Rayleigh, False) 
decBETA_erase_small_values = postprocess_evenlS(decTempBETA,L,J,listSize)

# We need to do a lot here: only preserve listSize for every section, others all zero
# Note down the listSize many indices and fading values in each section
decBetaSignificants = np.zeros((L, listSize) )
decBetaSignificantsPos = np.zeros((L, listSize), dtype=int )

for l in np.arange(L):
    for n in np.arange(listSize):
        decBeta_l = decBETA_erase_small_values[l*2**J : (l+1)*2**J]
        decBetaSignificants[l] = decBeta_l[decBeta_l!=0]
        decBetaSignificantsPos[l] = [pos for decBeta_l_element, pos in zip(decBeta_l,np.arange(len(decBeta_l))) if decBeta_l_element!=0]

decBetaSignificants = decBetaSignificants.transpose() # shape is (listSize, 16)
decBetaSignificantsPos = decBetaSignificantsPos.transpose() # shape is (listSize, 16)

tic = time.time()
# rxBits = Tree_decoder_uninformative_fading(decBetaSignificants, decBetaSignificantsPos, G,L,J, w, parityLengthVector,messageLengthVector,listSize)
rxBits, usedRootsIndex = Tree_decoder_fader(decBetaSignificants, decBetaSignificantsPos, L,J, w, parityLengthVector,messageLengthVector,listSize, parityDistribution)
toc = time.time()
print("Time of new algo " + str(toc-tic))

if rxBits.shape[0] > K: 
    rxBits = rxBits[np.arange(K)]

thisIter = 0
txBits_remained = np.empty(shape=(0,0))
for i in range(txBits.shape[0]):
    incre = 0
    incre = np.equal(txBits[i,:],rxBits).all(axis=1).any()
    thisIter += int(incre)
    if (incre == False):
        txBits_remained = np.vstack( (txBits_remained, txBits[i,:]) ) if txBits_remained.size else  txBits[i,:]
print("correctly recovers " + str(thisIter) + " out of " +str(rxBits.shape[0]) )



### To calculate genie
thisTimeGenie = 0
decOutMsg = convert_sparse_to_bits(decTempBETA, L, J, listSize, ) 

error_box = []
for i in range(txBitsParitized.shape[0]): # Recall txBitsParitized shape (100,256)
    flag_i = 1
    num_not_match_i = 0
    for l in range(L):                
        tmp = np.equal(txBitsParitized[i, l*J: (l+1)*J ], 
                        decOutMsg[:, l*J: (l+1)*J ]).all(axis=1).any()
        if tmp != True:
            flag_i = 0
            num_not_match_i += 1
    thisTimeGenie += flag_i
    if (flag_i ==0):
        error_box.append(num_not_match_i)
print("Genie recovers " + str(thisTimeGenie) +" out of " + str(K))
print(error_box)
print("error_box mean is " + str(np.mean(error_box))  )


rxBits_corrected = Tree_corrector_fader(decBetaSignificants, decBetaSignificantsPos, L,J, w, parityLengthVector,messageLengthVector,listSize, parityDistribution, usedRootsIndex)

print("corrected shape: " + str( rxBits_corrected.shape))


print("txBits_remained shape is :" + str(txBits_remained.shape))
corrected = 0
for i in range(txBits_remained.shape[0]):
    incre = 0
    incre = np.equal(txBits_remained[i,:],rxBits_corrected).all(axis=1).any()
    corrected += int(incre)
print("!!!!! CORRECTED " + str(corrected) + " out of " +str(rxBits_corrected.shape[0]) )


# tic = time.time()
# rxBits = Tree_decoder_uninformative(decOutMsg, G, L, J, w,
#                                                 parityLengthVector,
#                                                 messageLengthVector,
#                                                 listSize,)
# toc = time.time()
# print("time of old algo " + str(toc-tic))

# if rxBits.shape[0] > K: rxBits = rxBits[np.arange(K)]

# thisIter = 0
# for i in range(txBits.shape[0]):
#     incre = np.equal(txBits[i,:],rxBits).all(axis=1).any()
#     thisIter += incre
# print("Old decoder correctly recovers " + str(thisIter) + " out of " +str(rxBits.shape[0]) )