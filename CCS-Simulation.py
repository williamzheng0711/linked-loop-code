import ccsfg
import FactorGraphGeneration as FG
import ccsinnercode as ccsic
import numpy as np
from utils import *
from slow_lib import *
import tqdm

# Initialize CCS-AMP Graph
Graph = FG.Triadic8(16)

# Simulation Parameters for LDPC
Ka = 20                        # Number of active users
w = 128                        # Payload size of each active user (per user message length)
N = 38400                      # Total number of channel uses (real d.o.f)
listSize = Ka+2               # List size retained for each section after AMP converges
numAmpIter = 8                 # Number of AMP iterations
numBPIter = 2                  # Number of BP iterations to perform
BPonOuterGraph = True          # Indicates whether to perform BP on outer code.  If 0, AMP uses Giuseppe's uninformative prior
maxSims = 2                    # Number of Simulations to Run

EbNodB = 2.4                   # Energy per bit in decibels
EbNo = 10**(EbNodB/10)         # Eb/No in linear scale
P = 2*w*EbNo/N                 # transmit power
std = 1                        # Noise standard deviation
errorRate = 0.0                # track error rate across simulations



# Simulation Para for LLC
windowSize =2 
messageLen = 8
L = 16
Phat = N*P/L
J = 16
parityLen = J-messageLen
M = 2**J
Pa = parityLen*L
parityInvolved = get_parity_involvement_matrix(L,windowSize,messageLen)  
whichGMatrix = get_G_matrices(parityInvolved)



# Run CCS-AMP maxSims times
for idxsim in range(maxSims):
    print('Starting simulation %d of %d' % (idxsim + 1, maxSims))

    # # Generate random messages for Ka active users
    txBits = np.random.randint(low=2, size=(Ka, w))
    
    # Reset the graph
    Graph.reset()
    # Set up Inner Encoder/Decoder
    InnerCode = ccsic.BlockDiagonalInnerCode(N, P, std, Ka, Graph)
    # Outer LDPC Encoder
    txMsg = Graph.encodemessages(txBits)
    for msg in txMsg: 
        Graph.testvalid(msg)
    x = np.sum(txMsg, axis=0)
    # Inner CS Encoder
    x = InnerCode.Encode(x)
    # Transmit x over channel
    y = (x + (np.random.randn(N, 1) * std)).reshape(-1, 1)
    # Inner CS Decoder
    xHt, tau_evolution = InnerCode.Decode(y, numAmpIter, BPonOuterGraph, numBPIter, Graph)
    # Outer LDPC Decoder (Message Disambiguation)
    txMsgHt = Graph.decoder(xHt, listSize)
    # Calculate PUPE
    errorRate += (Ka - FG.numbermatches(txMsg, txMsgHt)) / (Ka * maxSims)


    encoded_tx_message = slow_encode(txBits,Ka,L,J,Pa,w,messageLen,parityLen,parityInvolved,whichGMatrix) 
    beta = convert_bits_to_sparse(encoded_tx_message,L,J,Ka)
    Ab, Az = sparc_codebook(L, M, N) 
    innerOutput = Ab(beta) 

    x = np.sqrt(Phat)*innerOutput                       # x is of size: (N, 1)
    z = np.random.randn(N, 1)                           # z is the Gaussian additive noise
    y = (x + z).reshape(-1, 1) 

    p0 = 1-(1-1/M)**Ka
    estimated_beta = amp_prior_art(y, P, L, M, numAmpIter, Ab, Az, p0)
    sigValues, sigPos = get_sig_values_and_positions(estimated_beta, L, J, listSize)
    print(" -Phase 1 (decoding) now starts.")
    # tic = time.time()
    chosenRoot = 0
    rxBits_llc, usedRootsIndex, listSizeOrder = slow_decoder(sigValues, sigPos, L, J, parityLen, messageLen, listSize, parityInvolved, whichGMatrix, windowSize, chosenRoot)
    # toc = time.time()
    # print(" | Time of LLC decode " + str(toc-tic))
    if rxBits_llc.shape[0] > Ka: 
        rxBits_llc = rxBits_llc[np.arange(Ka)]                    # As before, if we have >K paths, always choose the first K's.
    txBits_remained_llc = check_phase_1(txBits, rxBits_llc, "Linked-loop Code")

    rxBits_corrected_llc= slow_corrector(sigValues, sigPos,L,J,messageLen,parityLen,listSize,parityInvolved,usedRootsIndex,whichGMatrix,windowSize,listSizeOrder,chosenRoot)
    if txBits_remained_llc.shape[0] == w:
        txBits_remained_llc = txBits_remained_llc.reshape(1,-1)
    
    corrected = 0
    if rxBits_corrected_llc.size:
        for i in range(txBits_remained_llc.shape[0]):
            incre = 0
            incre = np.equal(txBits_remained_llc[i,:],rxBits_corrected_llc).all(axis=1).any()
            corrected += int(incre)
        print(" | In phase 2, Linked-loop code corrected " + str(corrected) + " true (one-outage) message out of " +str(rxBits_corrected_llc.shape[0]) )
    else: 
        print(" | Nothing was corrected")


# Display Simulation Results
print("Per user probability of error = %3.6f" % errorRate)
