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



def Tree_corrector_fader(decBetaNoised, decBetaPos,L,J,B,parityLengthVector,messageLengthVector,listSize, parityDistribution, usedRootsIndex):
    # decBetaSignificants size is (listSize, 16)
    cs_decoded_tx_message = np.zeros( (listSize, L*J) ) # (listSize, 256)
    for id_row in range(decBetaPos.shape[0]):
        for id_col in range(decBetaPos.shape[1]):
            a = np.binary_repr(decBetaPos[id_row][id_col], width=J)
            b = np.array([int(n) for n in a] ).reshape(1,-1)
            cs_decoded_tx_message[ id_row, id_col*J: (id_col+1)*J ] = b[0, 0:J]

    # decBetaNoised shape is (listSize, 16)
    listSizeOrder = np.argsort( decBetaNoised[:,0] )
    listSizeOrder_remained = [x for x in listSizeOrder if x not in usedRootsIndex]
    tree_decoded_tx_message = np.empty(shape=(0,0))

    for i, arg_i in zip(listSizeOrder_remained, np.arange(len(listSizeOrder_remained))):
        Paths = np.array([[i]])
        for l in range(1,L):
            new=np.empty( shape=(0,0))
            for j in range(Paths.shape[0]):
                Path=Paths[j].reshape(1,-1)
                # print("i=" + str(i) + " j=" + str(j) + " and Path is" + str(Path))
                # Compute the permissible parity check bits for the section
                Parity_computed = compute_permissible_parity_fader(Path,cs_decoded_tx_message,J,messageLengthVector, parityDistribution, l)
                # print("Parity_computed is: " + str(Parity_computed) )
                
                PathContainNa = len( np.where( Path[0] < 0 )[0] )
                containNaConditions = checkNaConditions(Path[0])

                noCandidates = True
                for k in range(listSize):
                    # Verify parity constraints for the children of surviving path
                    index = parity_check(Parity_computed,Path,k,cs_decoded_tx_message,L,J,parityLengthVector,messageLengthVector, parityDistribution)
                    # If parity constraints are satisfied, update the path
                    if index:
                        noCandidates = False
                        new = np.vstack((new,np.hstack((Path.reshape(1,-1),np.array([[k]]))))) if new.size else np.hstack((Path.reshape(1,-1),np.array([[k]])))
                    
                if noCandidates == True and containNaConditions and PathContainNa==0 :
                    new = np.vstack((new,np.hstack((Path.reshape(1,-1),np.array([[-1]]))))) if new.size else np.hstack((Path.reshape(1,-1),np.array([[-1]])))

            Paths = new 
            # print("Root No: "+ str(arg_i) + " l =" + str(l) + ". Path Shape is: " + str(Paths.shape[0]))

        if Paths.shape[0] >= 1: 
            # print("也许有correct!!!") 
            optimalOne = 0
            if Paths.shape[0] >= 2:
                pathVar = np.zeros((Paths.shape[0]))
                for whichPath in range(Paths.shape[0]):
                    fadingValues = []
                    for l in range(Paths.shape[1]): 
                        if Paths[whichPath][l] != -1:
                            fadingValues.append( decBetaNoised[ Paths[whichPath][l] ][l] )
                    pathVar[whichPath] = np.var(fadingValues)
                optimalOne = np.argmin(pathVar)

            onlyPathToConsider = Paths[optimalOne]
            sectionLost = np.where(onlyPathToConsider < 0)[0]
            decoded_message = np.zeros((1, L*J), dtype=int)
            for l in np.arange(L):
                if (l not in sectionLost):
                    decoded_message[0, l*J:(l+1)*J] = cs_decoded_tx_message[onlyPathToConsider[l], l*J:(l+1)*J]

            recovered_message = recover_msg(sectionLost, decoded_message, parityDistribution, messageLengthVector, J, L)
            tree_decoded_tx_message = np.vstack((tree_decoded_tx_message, recovered_message)) if tree_decoded_tx_message.size else recovered_message
    return tree_decoded_tx_message



def Tree_decoder_fader(decBetaNoised, decBetaPos, L,J,B,parityLengthVector,messageLengthVector,listSize, parityDistribution):
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

    for i, arg_i in zip(listSizeOrder, np.arange(len(listSizeOrder))):
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
                Parity_computed = compute_permissible_parity_fader(Path,cs_decoded_tx_message,J,messageLengthVector, parityDistribution, l)
                # print("Parity_computed is: " + str(Parity_computed) )
                for k in range(listSize):
                    # Verify parity constraints for the children of surviving path
                    index = parity_check(Parity_computed,Path,k,cs_decoded_tx_message,L,J,parityLengthVector,messageLengthVector, parityDistribution)
                    # If parity constraints are satisfied, update the path
                    if index:
                        new = np.vstack((new,np.hstack((Path.reshape(1,-1),np.array([[k]]))))) if new.size else np.hstack((Path.reshape(1,-1),np.array([[k]])))
            Paths = new 

            # print("Root No: "+ str(arg_i) + " l =" + str(l) + ". Path Shape is: " + str(Paths.shape[0]))

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


def checkNaConditions(Path):
    if len(Path)<16:
        Path_edited = np.zeros((16),dtype=int)
        Path_edited[0:len(Path)] = Path
    
    flag = True
    for elmt, idx in zip(Path_edited, np.arange(len(Path_edited))):
        if elmt >=0: continue
        else: 
            if (Path_edited[ (idx+1) % 16]<0 or Path_edited[ (idx+2) % 16]<0 or Path_edited[ (idx+3) % 16]<0 or Path_edited[ (idx+4) % 16]<0):
                flag = False

    return flag


def compute_permissible_parity_fader(Path,cs_decoded_tx_message,J,messageLengthVector, parityDistribution, currentl):
    # if path length  = 2
    # then we wanna have parity for section 2. section2Check = 2
    section2Check = Path.shape[1]
    parityDist = parityDistribution[:,section2Check].reshape(1,-1)[0]
    # print("----------parityDist: " + str(parityDist))
    parityDependents = np.nonzero(parityDist)[0]
    # print("parityDependents = " + str(parityDependents))
    Parity_computed = np.array([]).reshape(1,-1)      

    for l in parityDependents: 
        if l >= currentl: 
            continue
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
            toAppend = -1 * np.ones(1, parityDistribution[l,section2Check])[0]
        # print(toAppend)
        Parity_computed = np.concatenate((Parity_computed, toAppend), axis=None)
        # print("Parity_computed = " + str(Parity_computed))

    return Parity_computed


def recover_msg(sectionLost, decoded_message, parityDistribution, messageLengthVector, J, L):
    print("--- We will have a corrected message!! ---")
    # decoded_message is (1, L*J) = (1, 256)
    recovered_msg = np.array([], dtype= int).reshape(1,-1)
    for ll in np.arange(L):
        # print("ll=" +str(ll) )
        if ll not in sectionLost:
            recovered_msg = np.concatenate( (recovered_msg, decoded_message[0][ ll*J : ll*J+messageLengthVector[ll] ].reshape(1,-1)[0]) , axis=None )
        else: # ll  in sectoinLost:
            # suppose ll = 5. we first check section 5 determines what? 
            saverSections = np.nonzero(parityDistribution[ll])[0]
            # then saverSections = [6, 7, 8, 9]
            theLostPart = np.array([], dtype=int).reshape(1,-1)
            for l in saverSections:
                toAppend = decoded_message[0][ ( l*J + messageLengthVector[l] + int(sum(parityDistribution[0:ll,l])) ) :( l*J+messageLengthVector[l]+int(sum(parityDistribution[0:ll+1,l])) ) ].reshape(1,-1)[0]
                theLostPart = np.concatenate( (theLostPart, toAppend  )  , axis=None  )
            recovered_msg = np.concatenate( (recovered_msg, theLostPart) , axis=None)

    print("recovered_msg.shape = " + str(recovered_msg.shape))
    return recovered_msg.reshape(1,-1)