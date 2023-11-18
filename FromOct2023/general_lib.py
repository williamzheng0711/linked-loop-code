import numpy as np
import linkedloop as LLC

from general_utils import *
from tqdm import tqdm

def check_phase(txBits, rxBits, name, phase):
    # Check how many are correct amongst the recover (recover means first phase). No need to change.
    if len(rxBits) == 0:
        return txBits
    
    thisIter = 0
    txBits_remained = np.empty(shape=(0,0))
    for i in range(txBits.shape[0]):
        incre = 0
        incre = np.equal(txBits[i,:],rxBits).all(axis=1).any()
        thisIter += int(incre)
        if (incre == False):
            txBits_remained = np.vstack( (txBits_remained, txBits[i,:]) ) if txBits_remained.size else  txBits[i,:]
    print(" | In phase " + phase + " " + str(name) + " decodes " + str(thisIter) + " true message out of " +str(rxBits.shape[0]))
    # print(" - " + str(name) + " Phase 1 is done.")
    return txBits_remained


def partitioning_Gs(L, Gs, parityLens, windowSize):
    whichGMatrix = -1*np.ones((L,L), dtype=int)
    Gijs = {}
    for i in range(L):
        i_decides_who = np.mod( range(i+1, i+1+windowSize, 1), L)
        for j, idx in zip(i_decides_who, range(windowSize)):
            cipher = 2**i*3**j
            premable = sum(parityLens[i_decides_who[0:idx]])
            Gijs[cipher] = np.array(Gs[i])[:, premable:premable + parityLens[j]]
            whichGMatrix[i,j] = cipher
    return Gijs, whichGMatrix





def GLLC_UACE_decoder(rx_coded_symbols, L, J, Gijs, messageLens, parityLens, K, windowSize, whichGMatrix, SIC=True, pChosenRoot=None):

    # In decoder, will not change root
    cs_decoded_tx_message = -1 * np.ones((K, L*J))
    for id_row in range(K):
        for id_col in range(L):
            if rx_coded_symbols[id_row, id_col] != -1:
                a = np.binary_repr(rx_coded_symbols[id_row, id_col], width=J)     
                b = np.array([int(n) for n in a] ).reshape(1,-1)        
                cs_decoded_tx_message[id_row, id_col*J:(id_col+1)*J] = b[0,:]
    
    selected_cols = [l*J for l in range(L)]
    samples = cs_decoded_tx_message[:,selected_cols]
    num_erase = np.count_nonzero(samples == -1, axis=0) 

    K_effective  = [x for x in range(K) if cs_decoded_tx_message[x,0] != -1]
    tree_decoded_tx_message = np.empty(shape=(0,0))

    for i, _ in zip(K_effective, tqdm(range(len(K_effective)))):
        Paths = np.array([[i]])
        for l in list(range(1,L)): # its last element is L-1
            # print("l="+str(l) +" len=" + str(len(Paths)))
            new=np.empty( shape=(0,0))
            for Path in Paths:
                if l >= windowSize: 
                    Parity_computed = compute_parity(L=L, Path_list=Path, cs_decoded_tx_message=cs_decoded_tx_message, 
                                                            J=J, toCheck=l, whichGMatrix= whichGMatrix, messageLens=messageLens, 
                                                            parityLens=parityLens, Gijs=Gijs, windowSize=windowSize)
                for k in range(K):
                    index = l<windowSize or np.array_equal(cs_decoded_tx_message[k, l*J + messageLens[l]: (l+1)*J], Parity_computed)
                    if index:
                        new = np.vstack((new,np.hstack((Path.reshape(1,-1),np.array([[k]]))))) if new.size else np.hstack((Path.reshape(1,-1),np.array([[k]])))
            Paths = new
            if Paths.shape[0] == 0:
                break

        PathsUpdated = []
        for j in range(len(Paths)):
            Path = Paths[j]
            isOkay = final_parity_check(Path=Path, cs_decoded_tx_message=cs_decoded_tx_message, J=J,
                                        messageLens=messageLens, parityLens=parityLens, whichGMatrix=whichGMatrix, 
                                        L=L, Gijs=Gijs, windowSize=windowSize)
            if isOkay:
                PathsUpdated.append( Path )
        Paths = PathsUpdated
        # print("The root, surviving paths=" + str(len(Paths)))

        # print(len(Paths))
        if len(Paths) >= 1: # rows inside Paths should be all with one-outage. Some are true positive, some are false positive
            recovered_message = output_message(cs_decoded_tx_message, Paths, L, J, messageLens=messageLens)
            tree_decoded_tx_message = np.vstack((tree_decoded_tx_message, recovered_message)) if tree_decoded_tx_message.size else recovered_message
            # cancel the cdwd sections decoded, or cancel the used root
            pathToCancel = Paths[0]
            if SIC:
                for l in range(L):
                    if pathToCancel[l] != -1:
                        cs_decoded_tx_message[ pathToCancel[l], l*J:(l+1)*J] = -1*np.ones((J),dtype=int)
            else:
                if pChosenRoot == None: 
                    l = np.argmin(num_erase)
                    cs_decoded_tx_message[ pathToCancel[l], l*J:(l+1)*J] = -1*np.ones((J),dtype=int)
                else: 
                    cs_decoded_tx_message[ pathToCancel[pChosenRoot], pChosenRoot*J:(pChosenRoot+1)*J] = -1*np.ones((J),dtype=int)
    
    tree_decoded_tx_message = np.unique(tree_decoded_tx_message, axis=0)        
    return tree_decoded_tx_message, cs_decoded_tx_message, num_erase



def GLLC_UACE_corrector(cs_decoded_tx_message, L, J, Gs, Gijs, columns_index, sub_G_inversions, messageLens, parityLens, K, windowSize, whichGMatrix, num_erase, SIC=True, pChosenRoot=None):

    chosenRoot = np.argmin(num_erase) if pChosenRoot == None else pChosenRoot

    print(" | ChosenRoot: " + str(chosenRoot))

    messageLens[range(L)] = messageLens[np.mod(np.arange(chosenRoot, chosenRoot+L),L)]
    parityLens[range(L)] = parityLens[np.mod(np.arange(chosenRoot, chosenRoot+L),L)]
    num_erase[range(L)] = num_erase[np.mod(np.arange(chosenRoot, chosenRoot+L),L)]
    Gs[range(L)] = Gs[np.mod(np.arange(chosenRoot, chosenRoot+L),L)]
    columns_index[range(L)] = columns_index[np.mod(np.arange(chosenRoot, chosenRoot+L),L)]
    sub_G_inversions[range(L)] = sub_G_inversions[np.mod(np.arange(chosenRoot, chosenRoot+L),L)]
    whichGMatrix[:,range(L)] = whichGMatrix[:,np.mod(np.arange(chosenRoot, chosenRoot+L),L)]
    whichGMatrix[range(L),:] = whichGMatrix[np.mod(np.arange(chosenRoot, chosenRoot+L),L),:]
    cs_decoded_tx_message[:, range(L*J)] = cs_decoded_tx_message[:, np.mod( np.arange(chosenRoot*J, chosenRoot*J + L*J) ,L*J)]

    K_effective   = [x for x in range(K) if cs_decoded_tx_message[x,0] != -1]
    tree_decoded_tx_message = np.empty(shape=(0,0))

    for i, _ in zip(K_effective, tqdm(range(len(K_effective)))):
        Paths = [ LLC.GLinkedLoop([i], messageLens) ]
        for l in list(range(1,L)): # its last element is L-1
            if len(Paths) == 0: 
                break
            # print("l="+str(l) +" len=" + str(len(Paths)))
            newAll = []
            survivePaths = Parallel(n_jobs=-1)(delayed(GLLC_correct_each_section_and_path)(section2Check=l, Path=Paths[j], 
                                                                                           cs_decoded_tx_message=cs_decoded_tx_message, 
                                                                                           J=J, whichGMatrix=whichGMatrix, K=K, 
                                                                                           messageLens=messageLens, parityLens=parityLens, 
                                                                                           L=L, windowSize= windowSize, Gs=Gs, Gijs=Gijs, 
                                                                                           columns_index= columns_index,
                                                                                           sub_G_inversions= sub_G_inversions, 
                                                                                           num_erase = num_erase) for j in range(len(Paths)))
            # print("This section done.", end=" ")
            for survivePath in survivePaths:
                if len(survivePath) > 0:
                    newAll = list(newAll) + list(survivePath) # list merging
            Paths = newAll 

        # print("A root, before final checkng surviving paths=" + str(len(Paths)))
        # for path in Paths:
        #     if path.get_lostSection() == 1:
        #         print("***", path.get_path())


        PathsUpdated = []
        for j in range(len(Paths)):
            Path = Paths[j]
            isOkay = GLLC_final_parity_check(Path=Path, cs_decoded_tx_message=cs_decoded_tx_message, J=J,
                                             messageLens=messageLens, parityLens=parityLens, whichGMatrix=whichGMatrix, 
                                             L=L, Gijs=Gijs, windowSize=windowSize)
            if isOkay:
                PathsUpdated.append( Path )
        Paths = PathsUpdated
        # print("The root, surviving paths=" + str(len(Paths)))

        if len(Paths) >= 1: # rows inside Paths should be all with one-outage. Some are true positive, some are false positive
            # print(" | We obtained some candidate!!")
            Paths = [Paths[0]]
            recovered_message = GLLC_output_message(cs_decoded_tx_message, Paths, L, J)
            tree_decoded_tx_message = np.vstack((tree_decoded_tx_message, recovered_message)) if tree_decoded_tx_message.size else recovered_message
            if SIC:
                # for i in range(len(Paths)) and i==0:
                pathToCancel = Paths[0].get_path()
                # print(pathToCancel)
                for l in range(L):
                    if pathToCancel[l] != -1:
                        cs_decoded_tx_message[ pathToCancel[l], l*J:(l+1)*J] = -1*np.ones((J),dtype=int)
        
    w = sum(messageLens)
    tree_decoded_tx_message[:,range(w)] = tree_decoded_tx_message[:, np.mod( np.arange(w) + sum(messageLens[0:L-chosenRoot]), w)]
    # oldShape = tree_decoded_tx_message.shape
    tree_decoded_tx_message = np.unique(tree_decoded_tx_message, axis=0)
    # newShape = tree_decoded_tx_message.shape
    
    return tree_decoded_tx_message