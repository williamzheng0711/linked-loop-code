from numpy import pi, sqrt
import numpy as np
from scipy.special import erf
from scipy.stats import norm as normal
from scipy.stats import rice, rayleigh
from scipy.integrate import quad_vec
import matplotlib.pyplot as pyplot
import timeit
from scipy.linalg import hadamard
from utils import *

def Tree_corrector_fader(decBetaNoised, decBetaPos, G,L,J,B,parityLengthVector,messageLengthVector,listSize, parityDistribution, usedRootsIndex):
    # decBetaSignificants size is (listSize, 16)
    cs_decoded_tx_message = np.zeros( (listSize, L*J) ) # (listSize, 256)
    for id_row in range(decBetaPos.shape[0]):
        for id_col in range(decBetaPos.shape[1]):
            a = np.binary_repr(decBetaPos[id_row][id_col], width=J)
            # print("a = " + str(a))
            b = np.array([int(n) for n in a] ).reshape(1,-1)
            # print("b = " + str(b))
            cs_decoded_tx_message[ id_row, id_col*J: (id_col+1)*J ] = b[0, 0:J]

    # decBetaNoised shape is (listSize, 16)
    listSizeOrder = np.argsort( decBetaNoised[:,0] )
    listSizeOrder_remained = [x for x in listSizeOrder if x not in usedRootsIndex]
    tree_decoded_tx_message = np.empty(shape=(0,0))

    for i in listSizeOrder_remained:
        Paths = np.array([[i]])
        for l in range(1,L):
            # Grab the parity generator matrix corresponding to this section
            # G1 = G[l-1]
            new=np.empty( shape=(0,0))
            for j in range(Paths.shape[0]):
                Path=Paths[j].reshape(1,-1)
                # print("i=" + str(i) + " j=" + str(j) + " and Path is" + str(Path))
                # Compute the permissible parity check bits for the section
                Parity_computed = compute_permissible_parity_fader(Path,cs_decoded_tx_message,J,messageLengthVector, parityDistribution)
                # print("Parity_computed is: " + str(Parity_computed) )
                
                noCandidates = True
                for k in range(listSize):
                    # Verify parity constraints for the children of surviving path
                    index = parity_check(Parity_computed,Path,k,cs_decoded_tx_message,L,J,parityLengthVector,messageLengthVector)
                    # If parity constraints are satisfied, update the path
                    if index:
                        noCandidates = False
                        new = np.vstack((new,np.hstack((Path.reshape(1,-1),np.array([[k]]))))) if new.size else np.hstack((Path.reshape(1,-1),np.array([[k]])))
                    
                if noCandidates == True and PathContain0Na:
                    new = np.vstack((new,np.hstack((Path.reshape(1,-1),np.array([[-1]]))))) if new.size else np.hstack((Path.reshape(1,-1),np.array([[-1]])))

            Paths = new 

        if Paths.shape[0] >= 1:  
            if Paths.shape[0] >= 2:
                # If tree decoder outputs multiple paths for a root node, select the first one 
                flag = check_if_identical_msgs(Paths, cs_decoded_tx_message, L,J,parityLengthVector,messageLengthVector)
                if flag:
                    # print("Path[0] detail is " + str(Paths[0]))
                    tree_decoded_tx_message = np.vstack((tree_decoded_tx_message,extract_msg_bits(Paths[0].reshape(1,-1),cs_decoded_tx_message, L,J,parityLengthVector,messageLengthVector))) if tree_decoded_tx_message.size else extract_msg_bits(Paths[0].reshape(1,-1),cs_decoded_tx_message, L,J,parityLengthVector,messageLengthVector)
                else:
                    # print("Path shape is" + str(Paths.shape))
                    # print("Path[0] detail is " + str(Paths[0]))
                    optimalOne = 0
                    pathVar = np.zeros((Paths.shape[0]))
                    for whichPath in range(Paths.shape[0]):
                        fadingValues = []
                        for l in range(Paths.shape[1]):
                            # decBetaSignificantsPos size is (listSize, 16)s
                            fadingValues.append( decBetaNoised[ Paths[whichPath][l] ][l] )
                        
                        # print("fadingValues = " + str(fadingValues))
                        demeanFading = fadingValues - np.mean(fadingValues)
                        # pathVar[whichPath] = np.linalg.norm(demeanFading, 1)
                        pathVar[whichPath] = np.var(fadingValues)

                    optimalOne = np.argmin(pathVar)
                    tree_decoded_tx_message = np.vstack((tree_decoded_tx_message,extract_msg_bits(Paths[optimalOne].reshape(1,-1),cs_decoded_tx_message, L,J,parityLengthVector,messageLengthVector))) if tree_decoded_tx_message.size else extract_msg_bits(Paths[optimalOne].reshape(1,-1),cs_decoded_tx_message, L,J,parityLengthVector,messageLengthVector)
                    # tree_decoded_tx_message = np.vstack( (tree_decoded_tx_message,extract_msg_bits(Paths.reshape(Paths.shape[0],-1),cs_decoded_tx_message, L,J,parityLengthVector,messageLengthvector))) if tree_decoded_tx_message.size else extract_msg_bits(Paths.reshape(Paths.shape[0],-1),cs_decoded_tx_message, L,J,parityLengthVector,messageLengthvector)
            elif Paths.shape[0] == 1:
                tree_decoded_tx_message = np.vstack((tree_decoded_tx_message,extract_msg_bits(Paths.reshape(1,-1),cs_decoded_tx_message, L,J,parityLengthVector,messageLengthVector))) if tree_decoded_tx_message.size else extract_msg_bits(Paths.reshape(1,-1),cs_decoded_tx_message, L,J,parityLengthVector,messageLengthVector)


    return 1



def Tree_decoder_fader(decBetaNoised, decBetaPos, G,L,J,B,parityLengthVector,messageLengthVector,listSize, parityDistribution):
    # decBetaSignificants size is (listSize, 16)
    cs_decoded_tx_message = np.zeros( (listSize, L*J) ) # (listSize, 256)
    for id_row in range(decBetaPos.shape[0]):
        for id_col in range(decBetaPos.shape[1]):
            a = np.binary_repr(decBetaPos[id_row][id_col], width=J)
            # print("a = " + str(a))
            b = np.array([int(n) for n in a] ).reshape(1,-1)
            # print("b = " + str(b))
            cs_decoded_tx_message[ id_row, id_col*J: (id_col+1)*J ] = b[0, 0:J]

    # decBetaNoised shape is (listSize, 16)
    listSizeOrder = np.argsort( decBetaNoised[:,0] )
    tree_decoded_tx_message = np.empty(shape=(0,0))
    usedRootsIndex = []

    for i in listSizeOrder:
        Paths = np.array([[i]])
        for l in range(1,L):
            # Grab the parity generator matrix corresponding to this section
            # G1 = G[l-1]
            new=np.empty( shape=(0,0))
            for j in range(Paths.shape[0]):
                Path=Paths[j].reshape(1,-1)
                # print("i=" + str(i) + " j=" + str(j) + " and Path is" + str(Path))
                # Compute the permissible parity check bits for the section
                # Parity_computed = compute_permissible_parity(Path,cs_decoded_tx_message,G1,L,J,parityLengthVector,messageLengthvector)
                Parity_computed = compute_permissible_parity_fader(Path,cs_decoded_tx_message,J,messageLengthVector, parityDistribution)
                # print("Parity_computed is: " + str(Parity_computed) )
                for k in range(listSize):
                    # Verify parity constraints for the children of surviving path
                    index = parity_check(Parity_computed,Path,k,cs_decoded_tx_message,L,J,parityLengthVector,messageLengthVector)
                    # If parity constraints are satisfied, update the path
                    if index:
                        new = np.vstack((new,np.hstack((Path.reshape(1,-1),np.array([[k]]))))) if new.size else np.hstack((Path.reshape(1,-1),np.array([[k]])))
            Paths = new 

        if Paths.shape[0] >= 1:  
            if Paths.shape[0] >= 2:
                # If tree decoder outputs multiple paths for a root node, select the first one 
                flag = check_if_identical_msgs(Paths, cs_decoded_tx_message, L,J,parityLengthVector,messageLengthVector)
                if flag:
                    # print("Path[0] detail is " + str(Paths[0]))
                    tree_decoded_tx_message = np.vstack((tree_decoded_tx_message,extract_msg_bits(Paths[0].reshape(1,-1),cs_decoded_tx_message, L,J,parityLengthVector,messageLengthVector))) if tree_decoded_tx_message.size else extract_msg_bits(Paths[0].reshape(1,-1),cs_decoded_tx_message, L,J,parityLengthVector,messageLengthVector)
                else:
                    # print("Path shape is" + str(Paths.shape))
                    # print("Path[0] detail is " + str(Paths[0]))
                    optimalOne = 0
                    pathVar = np.zeros((Paths.shape[0]))
                    for whichPath in range(Paths.shape[0]):
                        fadingValues = []
                        for l in range(Paths.shape[1]):
                            # decBetaSignificantsPos size is (listSize, 16)s
                            fadingValues.append( decBetaNoised[ Paths[whichPath][l] ][l] )
                        
                        # print("fadingValues = " + str(fadingValues))
                        demeanFading = fadingValues - np.mean(fadingValues)
                        # pathVar[whichPath] = np.linalg.norm(demeanFading, 1)
                        pathVar[whichPath] = np.var(fadingValues)

                    optimalOne = np.argmin(pathVar)
                    tree_decoded_tx_message = np.vstack((tree_decoded_tx_message,extract_msg_bits(Paths[optimalOne].reshape(1,-1),cs_decoded_tx_message, L,J,parityLengthVector,messageLengthVector))) if tree_decoded_tx_message.size else extract_msg_bits(Paths[optimalOne].reshape(1,-1),cs_decoded_tx_message, L,J,parityLengthVector,messageLengthVector)
                    # tree_decoded_tx_message = np.vstack( (tree_decoded_tx_message,extract_msg_bits(Paths.reshape(Paths.shape[0],-1),cs_decoded_tx_message, L,J,parityLengthVector,messageLengthvector))) if tree_decoded_tx_message.size else extract_msg_bits(Paths.reshape(Paths.shape[0],-1),cs_decoded_tx_message, L,J,parityLengthVector,messageLengthvector)
            elif Paths.shape[0] == 1:
                tree_decoded_tx_message = np.vstack((tree_decoded_tx_message,extract_msg_bits(Paths.reshape(1,-1),cs_decoded_tx_message, L,J,parityLengthVector,messageLengthVector))) if tree_decoded_tx_message.size else extract_msg_bits(Paths.reshape(1,-1),cs_decoded_tx_message, L,J,parityLengthVector,messageLengthVector)

            usedRootsIndex.append(i)

    return tree_decoded_tx_message, usedRootsIndex



def compute_permissible_parity_fader(Path,cs_decoded_tx_message,J,messageLengthVector, parityDistribution):
    # if path length  = 2
    # then we wanna have parity for section 2. section2Check = 2
    section2Check = Path.shape[1]
    parityDist = parityDistribution[:,section2Check].reshape(1,-1)[0]
    # print("----------parityDist: " + str(parityDist))
    parityDependents = np.nonzero(parityDist)[0]
    # print("parityDependents = " + str(parityDependents))
    Parity_computed = np.array([]).reshape(1,-1)      

    for l in parityDependents: 
        # if section2Check = 2, l = 0, 1
        # j labels the sections we gonna check to fix section2Check's parities
        # cs_decoded_tx_message (listSize, 256)
        # print("l= " + str(l) + ", section2Check= " + str(section2Check))
        # print(l*J+sum(parityDistribution[l,0:section2Check]))
        # print(l*J+sum(parityDistribution[l,0:section2Check+1]))
        # print(Path)
        if (Path[0][l] != -1):
            toAppend = cs_decoded_tx_message[Path[0][l],l*J+sum(parityDistribution[l,0:section2Check]): l*J+sum(parityDistribution[l,0:section2Check+1])].reshape(1,-1)[0]
        elif Path[0][l] == -1:
            toAppend = -1 * np.ones(1, parityDistribution[l,0:section2Check])[0]
        # print(toAppend)
        Parity_computed = np.concatenate((Parity_computed, toAppend), axis=None)
        # print("Parity_computed = " + str(Parity_computed))

    return Parity_computed