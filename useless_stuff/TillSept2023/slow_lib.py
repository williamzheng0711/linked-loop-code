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
            parity_i = parity_i + (tx_message[:,decider*messageLen:(decider+1)*messageLen] @ generatorMatrices[useWhichMatrix[decider, i]])
        encoded_tx_message[:, i*J+parityLen:(i+1)*J] = np.mod(parity_i, 2)

    # One can check what a outer-encoded message looks like in the csv file.
    # np.savetxt('encoded_message.csv', encoded_tx_message[0].reshape(16,16), delimiter=',', fmt='%d')

    return encoded_tx_message

def slow_decoder(sigValues, sigPos, L, J, parityLen, messageLen, listSize, parityInvolved, whichGMatrix, windowSize):
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
    # Step 0: Preprocess
    losses = np.count_nonzero(sigPos == -1, axis=0) # losses is a L-long array
    # print(" UACE losses: " + str(losses))
    chosenRoot = np.argmax(losses)
    # chosenRoot = 0
    # print("chosenRoot: " + str(chosenRoot))
    sigPos[:,range(L)] = sigPos[:, np.mod(np.arange(chosenRoot, chosenRoot+L),L)]
    whichGMatrix[:,range(L)] = whichGMatrix[:,np.mod(np.arange(chosenRoot, chosenRoot+L),L)]
    whichGMatrix[range(L),:] = whichGMatrix[np.mod(np.arange(chosenRoot, chosenRoot+L),L),:]

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
    tree_decoded_tx_message = np.unique(tree_decoded_tx_message, axis=0)
    return tree_decoded_tx_message, np.concatenate((listSizeOrder[used_index], np.array(bad_roots, dtype=int)),axis=None), listSizeOrder


def slow_corrector(sigValues, sigPos, L, J, messageLen, parityLen, listSize, parityInvolved, usedRootsIndex, whichGMatrix, windowSize, listSizeOrder):
    
    cs_decoded_tx_message = -1* np.ones((listSize, L*J))
    for id_row in range(listSize):
        for id_col in range(L):
            if sigPos[id_row, id_col] != -1:
                a = np.binary_repr(sigPos[id_row, id_col], width=J)      # print("a = " + str(a))
                b = np.array([int(n) for n in a] ).reshape(1,-1)         # print("b = " + str(b))
                cs_decoded_tx_message[id_row, id_col*J:(id_col+1)*J] = b[0,:]

    listSizeOrder_remained = [x for x in listSizeOrder if x not in usedRootsIndex] # exclude used roots.
    tree_decoded_tx_message = np.empty(shape=(0,0))

    for i, _ in zip(listSizeOrder_remained, tqdm(range(len(listSizeOrder_remained)))):
        assert cs_decoded_tx_message[i,0] != -1
        Paths = [ LLC.LinkedLoop([i], messageLen) ]
        for l in list(range(1,L)): # its last element is L-1
            if len(Paths) == 0: 
                break
            newAll = []
            survivePaths = Parallel(n_jobs=-1)(delayed(slow_correct_each_section_and_path)(l,Paths[j],cs_decoded_tx_message,J, parityInvolved, 
                                                                                            whichGMatrix, listSize, messageLen, parityLen, L, windowSize) 
                                                                                        for j in range(len(Paths)))
            for survivePath in survivePaths:
                if len(survivePath) > 0:
                    newAll = list(newAll) + list(survivePath) # list merging
            Paths = newAll 

        PathsUpdated = []
        for j in range(len(Paths)):
            Path = Paths[j]
            isOkay = llc_final_parity_check(Path, cs_decoded_tx_message,J,messageLen,parityLen, parityInvolved, whichGMatrix, L)
            if isOkay:
                PathsUpdated.append( Path )
        Paths = PathsUpdated

        # For phase 2 correction, each root node at most give birth to ONE message corrected.
        if len(Paths) >= 1: # rows inside Paths should be all with one-outage. Some are true positive, some are false positive
            # print(" | We obtained some candidate!!")
            optimalOne = 0
            onlyPathToConsider = Paths[optimalOne]
            recovered_message = output_message(cs_decoded_tx_message, onlyPathToConsider, L, J)
            tree_decoded_tx_message = np.vstack((tree_decoded_tx_message, recovered_message)) if tree_decoded_tx_message.size else recovered_message

    # tree_decoded_tx_message[:,range(messageLen*L)] = tree_decoded_tx_message[:, np.mod( np.arange(messageLen*L)+(L-chosenRoot)*messageLen  , messageLen*L) ]
    tree_decoded_tx_message = np.unique(tree_decoded_tx_message, axis=0)

    return tree_decoded_tx_message









def llc_Aplus_decoder(sigValues, sigPos, L, J, parityLen, messageLen, listSize, parityInvolved, whichGMatrix, windowSize):
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
    """
    whichGMatrix1 = whichGMatrix.copy()

    # Step 0: Preprocess
    losses = np.count_nonzero(sigPos == -1, axis=0) # losses is a L-long array
    print("After A+ Channel, losses for each section are: " + str(losses))
    chosenRoot = np.argmax(losses)
    # chosenRoot = 0
    print("So choose root as section " + str(chosenRoot))
    sigPos[:,range(L)] = sigPos[:, np.mod(np.arange(chosenRoot, chosenRoot+L),L)]
    whichGMatrix[:,range(L)] = whichGMatrix[:,np.mod(np.arange(chosenRoot, chosenRoot+L),L)]
    whichGMatrix[range(L),:] = whichGMatrix[np.mod(np.arange(chosenRoot, chosenRoot+L),L),:]


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
    
    used_index = [a for a in range(len(results)) if np.sum(results[a][0]) >= 0 ]
    tree_decoded_tx_message = np.empty((0,0), dtype=int)
    for gd_idx in used_index:
        tree_decoded_tx_message = np.vstack((tree_decoded_tx_message,results[gd_idx])) if tree_decoded_tx_message.size else results[gd_idx]

    # 為了換位置
    tree_decoded_tx_message[:,range(messageLen*L)] = tree_decoded_tx_message[:, np.mod( np.arange(messageLen*L)+(L-chosenRoot)*messageLen  , messageLen*L)]
    tree_decoded_tx_message = np.unique(tree_decoded_tx_message, axis=0)

    whichGMatrix = whichGMatrix1
    return tree_decoded_tx_message






def llc_Aplus_corrector(sigPos, L, J, messageLen, parityLen, listSize, parityInvolved, whichGMatrix, windowSize, decodedCdwds, plus=True):

    cs_decoded_tx_message = -1* np.ones((listSize, L*J))
    for id_row in range(listSize):
        for id_col in range(L):
            if sigPos[id_row, id_col] != -1:
                a = np.binary_repr(sigPos[id_row, id_col], width=J)     
                b = np.array([int(n) for n in a] ).reshape(1,-1)        
                cs_decoded_tx_message[id_row, id_col*J:(id_col+1)*J] = b[0,:]

    cs_decoded_tx_message_test = cs_decoded_tx_message.copy()

    for decodedCdwd in decodedCdwds:
        for l in range(L):
            matches = ( cs_decoded_tx_message_test[: , l*J: l*J+messageLen] == decodedCdwd[l*J : l*J+messageLen] ).all(axis=1)
            matching_indices = np.where(matches)[0] # This must return something
            if len(matching_indices) > 0:
                cs_decoded_tx_message_test[ matching_indices[0], l*J:(l+1)*J] = -1*np.ones((J),dtype=int)

    selected_cols = [l*J for l in range(L)]
    samples = cs_decoded_tx_message_test[:,selected_cols]
    losses = np.count_nonzero(samples == -1, axis=0) # losses is a L-long array
    # print("losses in phase 2 " + str(losses))
    chosenRoot2 = np.argmin(losses)
    print("chosenRoot2: " + str(chosenRoot2))
    whichGMatrix[:,range(L)] = whichGMatrix[:,np.mod(np.arange(chosenRoot2, chosenRoot2+L),L)]
    whichGMatrix[range(L),:] = whichGMatrix[np.mod(np.arange(chosenRoot2, chosenRoot2+L),L),:]

    # for decodedCdwd in decodedCdwds:
    #     l = chosenRoot2
    #     matches = ( cs_decoded_tx_message[: , l*J: l*J+messageLen] == decodedCdwd[l*J : l*J+messageLen] ).all(axis=1)
    #     matching_indices = np.where(matches)[0] # This must return something
    #     # print(matching_indices)
    #     cs_decoded_tx_message[ matching_indices[0], l*J:(l+1)*J] = -1*np.ones((J),dtype=int)
    
    samples2 = cs_decoded_tx_message[:,selected_cols]
    losses2 = np.count_nonzero(samples2 == -1, axis=0) # losses is a L-long array
    print("losses in phase 2 " + str(losses2))
    
    cs_decoded_tx_message[:, range(L*J)] = cs_decoded_tx_message[:, np.mod( np.arange(chosenRoot2*J, chosenRoot2*J + L*J) ,L*J)]


    K_remained   = [x for x in range(listSize) if cs_decoded_tx_message[x,0] != -1]
    tree_decoded_tx_message = np.empty(shape=(0,0))

    for i, _ in zip(K_remained, tqdm(range(len(K_remained)))):
        # assert cs_decoded_tx_message[i,0] != -1
        Paths = [ LLC.LinkedLoop([i], messageLen) ]
        for l in list(range(1,L)): # its last element is L-1
            if len(Paths) == 0: 
                break
            newAll = []
            survivePaths = Parallel(n_jobs=-1)(delayed(slow_correct_each_section_and_path)(l,Paths[j],cs_decoded_tx_message,J, parityInvolved, 
                                                                                            whichGMatrix, listSize, messageLen, parityLen, L, windowSize) 
                                                                                        for j in range(len(Paths)))
            for survivePath in survivePaths:
                if len(survivePath) > 0:
                    newAll = list(newAll) + list(survivePath) # list merging
            Paths = newAll 

        PathsUpdated = []
        for j in range(len(Paths)):
            Path = Paths[j]
            isOkay = llc_final_parity_check(Path, cs_decoded_tx_message,J,messageLen,parityLen, parityInvolved, whichGMatrix, L)
            if isOkay:
                PathsUpdated.append( Path )
        Paths = PathsUpdated


        if len(Paths) >= 1: # rows inside Paths should be all with one-outage. Some are true positive, some are false positive
            # print(" | We obtained some candidate!!")
            recovered_message = output_message(cs_decoded_tx_message, Paths, L, J)
            tree_decoded_tx_message = np.vstack((tree_decoded_tx_message, recovered_message)) if tree_decoded_tx_message.size else recovered_message
            # SIC
            if plus:
                for i in range(len(Paths)):
                    pathToCancel = Paths[i].get_path()
                    for l in range(L):
                        if pathToCancel[l] != -1:
                            cs_decoded_tx_message[ pathToCancel[l], l*J:(l+1)*J] = -1*np.ones((J),dtype=int)
            
        # samples = cs_decoded_tx_message[:,selected_cols]
        # losses = np.count_nonzero(samples == -1, axis=0) # losses is a L-long array
        # print("losses " + str(losses))
        

    tree_decoded_tx_message[:,range(messageLen*L)] = tree_decoded_tx_message[:, np.mod( np.arange(messageLen*L)+(L-chosenRoot2)*messageLen  , messageLen*L)]
    tree_decoded_tx_message = np.unique(tree_decoded_tx_message, axis=0)
    
    return tree_decoded_tx_message




def llc_UACE_decoder(sigPos, L, J, messageLen, parityLen, listSize, parityInvolved, whichGMatrix, windowSize, APlus=True):

    cs_decoded_tx_message = -1 * np.ones((listSize, L*J))
    for id_row in range(listSize):
        for id_col in range(L):
            if sigPos[id_row, id_col] != -1:
                a = np.binary_repr(sigPos[id_row, id_col], width=J)     
                b = np.array([int(n) for n in a] ).reshape(1,-1)        
                cs_decoded_tx_message[id_row, id_col*J:(id_col+1)*J] = b[0,:]

    selected_cols = [l*J for l in range(L)]
    samples = cs_decoded_tx_message[:,selected_cols]
    num_erase = np.count_nonzero(samples == -1, axis=0) 
    chosenRoot = np.argmin(num_erase)
    print(" | Num erase: " + str(num_erase))
    print(" | ChosenRoot: " + str(chosenRoot))

    whichGMatrix[:,range(L)] = whichGMatrix[:,np.mod(np.arange(chosenRoot, chosenRoot+L),L)]
    whichGMatrix[range(L),:] = whichGMatrix[np.mod(np.arange(chosenRoot, chosenRoot+L),L),:]
    cs_decoded_tx_message[:, range(L*J)] = cs_decoded_tx_message[:, np.mod( np.arange(chosenRoot*J, chosenRoot*J + L*J) ,L*J)]

    K_remained   = [x for x in range(listSize) if cs_decoded_tx_message[x,0] != -1]
    tree_decoded_tx_message = np.empty(shape=(0,0))

    for i, _ in zip(K_remained, tqdm(range(len(K_remained)))):
        # assert cs_decoded_tx_message[i,0] != -1
        Paths = [ LLC.LinkedLoop([i], messageLen) ]
        for l in list(range(1,L)): # its last element is L-1
            if len(Paths) == 0: 
                break
            newAll = []
            survivePaths = Parallel(n_jobs=-1)(delayed(slow_correct_each_section_and_path)(l,Paths[j],cs_decoded_tx_message,J, parityInvolved, 
                                                                                            whichGMatrix, listSize, messageLen, parityLen, L, windowSize) 
                                                                                        for j in range(len(Paths)))
            for survivePath in survivePaths:
                if len(survivePath) > 0:
                    newAll = list(newAll) + list(survivePath) # list merging
            Paths = newAll 

        PathsUpdated = []
        for j in range(len(Paths)):
            Path = Paths[j]
            isOkay = llc_final_parity_check(Path, cs_decoded_tx_message,J,
                                            messageLen,parityLen, parityInvolved, 
                                            whichGMatrix, L, True)
            if isOkay:
                PathsUpdated.append( Path )
        Paths = PathsUpdated

        if len(Paths) >= 1: # rows inside Paths should be all with one-outage. Some are true positive, some are false positive
            # print(" | We obtained some candidate!!")
            recovered_message = output_message(cs_decoded_tx_message, Paths, L, J)
            tree_decoded_tx_message = np.vstack((tree_decoded_tx_message, recovered_message)) if tree_decoded_tx_message.size else recovered_message
            # SIC
            if APlus:
                for i in range(len(Paths)):
                    pathToCancel = Paths[i].get_path()
                    print(pathToCancel)
                    for l in range(L):
                        if pathToCancel[l] != -1:
                            cs_decoded_tx_message[ pathToCancel[l], l*J:(l+1)*J] = -1*np.ones((J),dtype=int)
            
        # samples = cs_decoded_tx_message[:,selected_cols]
        # losses = np.count_nonzero(samples == -1, axis=0) # losses is a L-long array
        # print("losses " + str(losses))
        

    tree_decoded_tx_message[:,range(messageLen*L)] = tree_decoded_tx_message[:, np.mod( np.arange(messageLen*L)+(L-chosenRoot)*messageLen  , messageLen*L)]
    tree_decoded_tx_message = np.unique(tree_decoded_tx_message, axis=0)
    
    return tree_decoded_tx_message