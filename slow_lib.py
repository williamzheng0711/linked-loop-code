import numpy as np
from utils import *
from slow_utils import *
from joblib import Parallel, delayed

def slow_encode(tx_message, K, L, J, Pa, Ml, messageLengthVector, parityLengthVector, parityDistribution, useWhichMatrix):
    """
    Encode tx_message ( of size (K,w) ) into (K, 2*w). 
    Each row, aka each message by an user, is paritized hence longer.

    Parameters
    ----------
    tx_message (ndarray): K x w matrix of K users' w-bit messages
    K (int): number of active users
    L (int): number of sections in codeword
    J (int): number of bits/section
    Pa (int): total number of parity bits
    Ml (int): total number of message bits
    messageLengthVector (ndarray): 1 x L vector indicating # message bits/section
    parityLengthVector (ndarray): 1 x L vector indicating # parity bits/section
    parityDistribution (ndarray): L x L matrix of info/parity bit connections
    useWhichMatrix (ndarray): L x L matrix indicating which generator to use 

    Returns
    -------
    encoded_tx_message : ndarray (K by (w+Pa) matrix, or 100 by 256 in usual case)
    """

    encoded_tx_message = np.zeros((K, Ml+Pa), dtype=int)
    m = messageLengthVector[0]
    generatorMatrices = matrix_repo(dim=8)
    for i in range(L):
        encoded_tx_message[:,i*J:i*J+m] = tx_message[:,i*m:(i+1)*m]
        whoDecidesI = np.where(parityDistribution[:, i])[0]
        parity_i = np.zeros((K, m), dtype=int)
        for decider in whoDecidesI:
            parity_i += (tx_message[:,decider*m:(decider+1)*m] @ generatorMatrices[useWhichMatrix[decider, i]])
        encoded_tx_message[:, i*J+m:(i+1)*J] = np.mod(parity_i, 2)

    # One can check what a outer-encoded message looks like in the csv file.
    # np.savetxt('encoded_message.csv', encoded_tx_message[0].reshape(16,16), delimiter=',', fmt='%d')

    return encoded_tx_message

def slow_decoder(sigValues, sigPos, L, J, w, parityLengthVector, messageLengthVector, listSize, parityInvolved, whichGMatrix):
    """
    Phase 1 decoder (no erasure correction)

        Arguments:
            sigValues (ndarray): listSize x L matrix of significant values per section of recovered codeword
            sigPos (ndarray): listSize x L matrix of positions of significant values in recovered codeword
            L (int): number of sections in recovered codeword
            J (int): number of bits/section in codeword
            w (int): number of information bits
            parityLengthVector (ndarray): number of parity bits/section
            messageLengthVector (ndarray): number of information bits/section
            listSize (int): number of entries to retain per section in recovered codeword
            parityInvolved (ndarray): indicator matrix of parity to information section connections
            whichGMatrix (ndarray): matrix indicating which generator matrix connects parity to info sections

        Returns:
            tree_decoded_tx_message (ndarray): decoded messages
            usedRootsIndex (ndarray): indices of roots that lead to parity consistent paths  
    """
    # Step 1: reconstruct L lists of listSize message fragments
    cs_decoded_tx_message = np.zeros((listSize, L*J))
    for id_row in range(listSize):
        for id_col in range(L):
            a = np.binary_repr(sigPos[id_row, id_col], width=J)      # print("a = " + str(a))
            b = np.array([int(n) for n in a] ).reshape(1,-1)         # print("b = " + str(b))
            cs_decoded_tx_message[id_row, id_col*J:(id_col+1)*J] = b[0,:]

    # Step 2: find parity consistent paths
    listSizeOrder = np.argsort(sigValues[:, 0])[::-1]
    results = Parallel(n_jobs=-1)(delayed(slow_decode_deal_with_root_i)(idx, L, cs_decoded_tx_message, J, parityInvolved, whichGMatrix, messageLengthVector, listSize, parityLengthVector, sigValues) for idx in listSizeOrder) 
    results = np.array(results).squeeze()
    flag_good_results = (np.sum(results, axis=1) > 0).astype(int)
    idx_good_results = np.where(flag_good_results)[0]
    tree_decoded_tx_message = results[idx_good_results, :]
    return tree_decoded_tx_message, listSizeOrder[idx_good_results]

def slow_corrector(sigValues, sigPos, L, J, B, parityLengthVector, messageLengthVector, listSize, parityInvolved, usedRootsIndex, whichGMatrix):
    cs_decoded_tx_message = np.zeros( (listSize, L*J) )
    for id_row in range(sigPos.shape[0]):
        for id_col in range(sigPos.shape[1]):
            a = np.binary_repr(sigPos[id_row][id_col], width=J)
            b = np.array([int(n) for n in a] ).reshape(1,-1)
            cs_decoded_tx_message[ id_row, id_col*J: (id_col+1)*J ] = b[0, 0:J]

    listSizeOrder = np.flip(np.argsort( sigValues[:,0] )) 
    listSizeOrder_remained = [x for x in listSizeOrder if x not in usedRootsIndex] # exclude used roots.
    tree_decoded_tx_message = np.empty(shape=(0,0))
    targetingSections = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,0]

    for i, arg_i in zip(listSizeOrder_remained, np.arange(len(listSizeOrder_remained))):
        Paths = np.array([[i]])
        for l, arg_l in zip( targetingSections, range(len(targetingSections))):
            # print( "Target section: " + str(l) + " | No. of paths: " + str(Paths.shape[0]) + " | How many contains -1: " + str(sum([1 for Path in Paths if np.any(Path<0)])) )
            if Paths.shape[1] == 0: 
                print("-------")
                break

            newAll=np.empty( shape=(0,0))
            if l!=0 :  # We still need to enlarge lenth of Paths.
                
                survivePaths = Parallel(n_jobs=-1)(delayed(slow_correct_each_section_and_path)(l, j, Paths, cs_decoded_tx_message, J, parityInvolved, whichGMatrix, listSize, messageLengthVector) for j in range(Paths.shape[0]))
                for survivePath in survivePaths:
                    if survivePath.size:
                        newAll = np.vstack((newAll, survivePath)) if newAll.size else survivePath

                Paths = newAll 
            
            else: # We dont enlarge length of Paths anymore
                PathsUpdated = np.empty( shape=(0,0))
                for j in range(Paths.shape[0]):
                    isOkay = False
                    Path = Paths[j].reshape(1,-1)
                    isOkay = slow_parity_check( None, Path, None, cs_decoded_tx_message,J,messageLengthVector, parityInvolved, whichGMatrix)
                    if isOkay:
                        PathsUpdated = np.vstack((PathsUpdated, Path)) if PathsUpdated.size else Path
                Paths = PathsUpdated

        if Paths.shape[0] >= 1: # rows inside Paths should be all with one-outage. Some are true positive, some are false positive
            print(" | We obtained some candidate!!")
            optimalOne = 0
            if Paths.shape[0] >= 2:
                pathVar = np.zeros((Paths.shape[0]))
                for whichPath in range(Paths.shape[0]):
                    fadingValues = []
                    for l in range(Paths.shape[1]): 
                        if Paths[whichPath][l] != -1:
                            fadingValues.append( sigValues[ Paths[whichPath][l] ][l] )
                    pathVar[whichPath] = np.var(fadingValues)
                optimalOne = np.argmin(pathVar)

            onlyPathToConsider = Paths[optimalOne]
            sectionLost = np.where(onlyPathToConsider < 0)[0]
            decoded_message = np.zeros((1, L*J), dtype=int)
            for l in np.arange(L):
                if (l not in sectionLost):
                    decoded_message[0, l*J:(l+1)*J] = cs_decoded_tx_message[onlyPathToConsider[l], l*J:(l+1)*J]

            recovered_message = slow_recover_msg(sectionLost, decoded_message, parityInvolved, messageLengthVector, J, L, whichGMatrix)
            if recovered_message != np.array([], dtype= int).reshape(1,-1):
                tree_decoded_tx_message = np.vstack((tree_decoded_tx_message, recovered_message)) if tree_decoded_tx_message.size else recovered_message

    return tree_decoded_tx_message

def slow_correct_each_section_and_path(l, j, Paths, cs_decoded_tx_message, J, parityInvolved, whichGMatrix, listSize, messageLengthVector):
    new = np.empty( shape=(0,0), dtype=int)
    Path = Paths[j].reshape(1,-1)
    pathArgNa = np.where( Path[0] < 0 )[0]    
    Parity_computed = -1 * np.ones((1,8),dtype=int)
    if l >= 4: 
        Parity_computed = slow_compute_permissible_parity(Path, cs_decoded_tx_message, J, parityInvolved, l, whichGMatrix)
    for k in range(listSize):
        if l < 4:
            new = np.vstack((new,np.hstack((Path.reshape(1,-1),np.array([[k]]))))) if new.size else np.hstack((Path.reshape(1,-1),np.array([[k]])))
        else :  # now l >= 4:
            index = slow_parity_check(Parity_computed, Path, k, cs_decoded_tx_message, J, messageLengthVector, parityInvolved, whichGMatrix) 
            if index:
                new = np.vstack((new,np.hstack((Path.reshape(1,-1),np.array([[k]]))))) if new.size else np.hstack((Path.reshape(1,-1),np.array([[k]])))
    if len(pathArgNa) == 0:
        new = np.vstack((new,np.hstack((Path.reshape(1,-1),np.array([[-1]]))))) if new.size else np.hstack((Path.reshape(1,-1),np.array([[-1]])))

    return new