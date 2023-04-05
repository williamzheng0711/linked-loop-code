import numpy as np
from utils import *
from slow_utils import *
from joblib import Parallel, delayed
from tqdm import tqdm
import linkedloop as LLC
    

def slow_encode(tx_message, K, L, J, Pa, w, messageLen, parityLen, parityDistribution, useWhichMatrix):
    """
    Encode tx_message ( of size (K,w) ) into (K, w + Pa). 
    Each row, aka each message by an user, is paritized hence longer.

    Parameters
    ----------
    tx_message (ndarray): K x w matrix of K users' w-bit messages
    K (int): number of active users
    L (int): number of sections in codeword
    J (int): number of bits/section
    Pa (int): total number of parity bits
    w (int): total number of message bits
    messageLen (int): number of info bits in each section
    parityLen (int): number of parity bits in each section (L = parityLen + messageLen)
    parityDistribution (ndarray): L x L matrix of info/parity bit connections
    useWhichMatrix (ndarray): L x L matrix indicating which generator to use 

    Returns
    -------
    encoded_tx_message : ndarray (K by (w+Pa) matrix, or 100 by 256 in usual case)
    """

    encoded_tx_message = np.zeros((K, w+Pa), dtype=int)
    generatorMatrices = matrix_repo(dim=messageLen)
    for i in range(L):
        encoded_tx_message[:,i*J:i*J+messageLen] = tx_message[:,i*messageLen:(i+1)*messageLen]
        whoDecidesI = np.where(parityDistribution[:, i])[0]
        parity_i = np.zeros((K, parityLen), dtype=int)
        for decider in whoDecidesI:
            parity_i += (tx_message[:,decider*messageLen:(decider+1)*messageLen] @ generatorMatrices[useWhichMatrix[decider, i]])
        encoded_tx_message[:, i*J+parityLen:(i+1)*J] = np.mod(parity_i, 2)

    # One can check what a outer-encoded message looks like in the csv file.
    # np.savetxt('encoded_message.csv', encoded_tx_message[0].reshape(16,16), delimiter=',', fmt='%d')

    return encoded_tx_message

def slow_decoder(sigValues, sigPos, L, J, parityLen, messageLen, listSize, parityInvolved, whichGMatrix, windowSize, chosenRoot):
    """
    Phase 1 decoder (no erasure correction)

        Arguments:
            sigValues (ndarray): listSize x L matrix of significant values per section of recovered codeword
            sigPos (ndarray): listSize x L matrix of positions of significant values in recovered codeword
            L (int): number of sections in recovered codeword
            J (int): number of bits/section in codeword
            messageLen (int): number of info bits/section in codeword
            listSize (int): number of entries to retain per section in recovered codeword
            parityInvolved (ndarray): indicator matrix of parity to information section connections
            whichGMatrix (ndarray): matrix indicating which generator matrix connects parity to info sections
            windowSize (int): number of previous consecutive sections needed to calculate a section's parity portion

        Returns:
            tree_decoded_tx_message (ndarray): decoded messages
            usedRootsIndex (ndarray): indices of roots that lead to parity consistent paths  
            listSizeOrder 
    """
    # Step 1: reconstruct L lists of listSize message fragments
    bad_roots = []
    cs_decoded_tx_message = -1* np.ones((listSize, L*J))
    for id_row in range(listSize):
        for id_col in range(L):
            if sigPos[id_row, id_col] != -1:
                a = np.binary_repr(sigPos[id_row, id_col], width=J)      # print("a = " + str(a))
                b = np.array([int(n) for n in a] ).reshape(1,-1)         # print("b = " + str(b))
                cs_decoded_tx_message[id_row, id_col*J:(id_col+1)*J] = b[0,:]
            elif id_col == 0:
                if id_row not in bad_roots:
                    bad_roots.append(id_row)

    listSizeOrder = np.argsort(sigValues[:, 0])[::-1]

    # Step 2: find parity consistent paths    
    results = Parallel(n_jobs=-1)(delayed(slow_decode_deal_with_root_i)
                                  (idx, L, cs_decoded_tx_message, J, parityInvolved, whichGMatrix, messageLen, listSize, parityLen, windowSize) 
                                  for idx in listSizeOrder)     
    
    used_index = [a for a in range(len(results)) if sum(np.sum(results[a],axis=1)) >=0]
    tree_decoded_tx_message = np.empty((0,0), dtype=int)
    for gd_idx in used_index:
        tree_decoded_tx_message = np.vstack((tree_decoded_tx_message,results[gd_idx])) if tree_decoded_tx_message.size else results[gd_idx]

    tree_decoded_tx_message[:,range(messageLen*L)] = tree_decoded_tx_message[:, np.mod( np.arange(messageLen*L)+(L-chosenRoot)*messageLen  , messageLen*L) ]
    return tree_decoded_tx_message, np.concatenate((listSizeOrder[used_index], np.array(bad_roots, dtype=int)),axis=None), listSizeOrder


def slow_corrector(sigValues, sigPos, L, J, messageLen, parityLen, listSize, parityInvolved, usedRootsIndex, whichGMatrix, windowSize, listSizeOrder, chosenRoot):
    cs_decoded_tx_message = np.zeros( (listSize, L*J) )
    for id_row in range(sigPos.shape[0]):
        for id_col in range(sigPos.shape[1]):
            a = np.binary_repr(sigPos[id_row][id_col], width=J)
            b = np.array([int(n) for n in a] ).reshape(1,-1)
            cs_decoded_tx_message[ id_row, id_col*J: (id_col+1)*J ] = b[0, 0:J]
    
    listSizeOrder_remained = [x for x in listSizeOrder if x not in usedRootsIndex] # exclude used roots.
    tree_decoded_tx_message = np.empty(shape=(0,0))
    targetingSections = np.mod(np.arange(1,L+1),L)

    for i, idx in zip(listSizeOrder_remained, tqdm(range(len(listSizeOrder_remained)))):
        assert cs_decoded_tx_message[i,0] != -1
        Paths = np.array([ LLC.LinkedLoop(np.array([i]), messageLen, -1*np.ones(messageLen, dtype=int)) ]) # Paths is still an nparray, its elements are LinkedLoops
        for l in targetingSections:
            # print( "Target section: " + str(l) + " | No. of paths: " + str(Paths.shape[0]) + " | How many contains -1: " + str(sum([1 for Path in Paths if np.any(Path<0)])) )
            if Paths.shape[0] == 0: 
                break
            newAll=np.empty( shape=(0,0))
            
            if l != 0 :  # We still need to enlarge lenth of Paths.
                survivePaths = Parallel(n_jobs=-1)(delayed(slow_correct_each_section_and_path)(l, Paths[j], cs_decoded_tx_message, J, 
                                                                                               parityInvolved, whichGMatrix, listSize, 
                                                                                               messageLen, parityLen, L, windowSize) 
                                                                                            for j in range(Paths.shape[0]))
                for survivePath in survivePaths:
                    if survivePath.size:
                        newAll = np.hstack((newAll, survivePath)) if newAll.size else survivePath
                Paths = newAll 

            else: # We dont enlarge length of Paths anymore
                PathsUpdated = np.empty( shape=(0), dtype=object)
                for j in range(Paths.shape[0]):
                    Path = Paths[j]
                    isOkay = llc_final_parity_check(Path, cs_decoded_tx_message,J,messageLen,parityLen, parityInvolved, whichGMatrix, L)
                    if isOkay:
                        PathsUpdated = np.append(PathsUpdated, Path)
                Paths = PathsUpdated


        # For phase 2 correction, each root node at most give birth to ONE message corrected.
        if Paths.shape[0] >= 1: # rows inside Paths should be all with one-outage. Some are true positive, some are false positive
            # print(" | We obtained some candidate!!")
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

            # sectionLost = np.where(onlyPathToConsider < 0)[0]
            # decoded_message = np.zeros((1, L*J), dtype=int)
            # for l in np.arange(L):
            #     if (l not in sectionLost):
            #         decoded_message[0, l*J:(l+1)*J] = cs_decoded_tx_message[onlyPathToConsider[l], l*J:(l+1)*J]
            # recovered_message = slow_recover_msg(sectionLost, decoded_message, parityInvolved, messageLen,parityLen, J, L, whichGMatrix)
            
            recovered_message = output_message(cs_decoded_tx_message, onlyPathToConsider, L, J)

            # if recovered_message != np.array([], dtype= int).reshape(1,-1):
            tree_decoded_tx_message = np.vstack((tree_decoded_tx_message, recovered_message)) if tree_decoded_tx_message.size else recovered_message

    tree_decoded_tx_message[:,range(messageLen*L)] = tree_decoded_tx_message[:, np.mod( np.arange(messageLen*L)+(L-chosenRoot)*messageLen  , messageLen*L) ]
    return tree_decoded_tx_message


