import numpy as np
import linkedloop as LLC
import time

from general_utils import *
from static_repo import *
from abch_utils import *
from tqdm import tqdm
from joblib import Parallel, delayed


### We use this file to store functions/methods that are inside the launcher and related to decoding procedure.

def encode(tx_message,K,L,N,M,messageLens,parityLens, Gijs):
    """
    Parameters
    ----------
    tx_message (ndarray): K x B matrix of K users' B-bit messages
    K (int): number of active users
    L (int): number of sections in codeword
    N (int): number of bits in codeword

    Returns
    -------
    encoded_tx_message : ndarray (K by N matrix, or 100 by 256 in usual case)
    """
    encoded_tx_message = np.zeros((K, N), dtype=int)
    for l in range(L):
        encoded_tx_message[:,l*J:l*J+messageLens[l]] = tx_message[:, sum(messageLens[0:l]): sum(messageLens[0:l])+ messageLens[l]]
        who_decides_pl = who_decides_p_sec(L,l,M)
        parity_l = np.zeros((K, parityLens[l]), dtype=int)
        for decider in who_decides_pl: 
            toAdd= (tx_message[:,sum(messageLens[0:decider]):sum(messageLens[0:decider])+ messageLens[decider]] @ Gijs[CantorPairing(decider,l)] )
            parity_l= parity_l+ toAdd
        encoded_tx_message[: ,l*J+messageLens[l]: (l+1)*J]= np.mod(parity_l, 2)
    # One can check what a outer-encoded message looks like in the csv file.
    # np.savetxt('encoded_message.csv', encoded_tx_message[0].reshape(16,16), delimiter=',', fmt='%d')
    return encoded_tx_message


def phase1_decoder(grand_list, L, Gijs, messageLens, parityLens, K, M, SIC=True, pChosenRoot=None):
    """
    Parameters
    ----------
    K (int): number of active users
    L (int): number of sections in codeword
    N (int): number of bits in codeword

    Returns
    -------
    encoded_tx_message : ndarray (K by N matrix, or 100 by 256 in usual case)
    """

    # selected_cols = [l*J for l in range(L)]
    # samples = grand_list[:,selected_cols]
    # num_erase = np.count_nonzero(samples == -1, axis=0) 
    K_effective  = [x for x in range(K) if grand_list[x,0] != -1]
    decoded_msg = np.empty(shape=(0,0))

    for i, _ in zip(K_effective, tqdm(range(len(K_effective)))):
        Paths = np.array([[i]])
        for l in list(range(1,L)):
            new= np.empty(shape= (0, 0))
            for Path in Paths:
                if l >= M: 
                    pl_computed = compute_parity(L, Path, grand_list, l, messageLens, parityLens, Gijs, M)
                for k in range(K):
                    index= (l< M) or np.array_equal(grand_list[k, l*J + messageLens[l]: (l+1)*J], pl_computed)
                    if index:
                        new = np.vstack((new,np.hstack((Path.reshape(1,-1),np.array([[k]]))))) if new.size else np.hstack((Path.reshape(1,-1),np.array([[k]])))
            Paths = new
            if Paths.shape[0] == 0: break

        Paths2Return = []
        for j in range(len(Paths)):
            Path = Paths[j]
            pathOk = final_parity_check(Path, grand_list, messageLens, parityLens, L, Gijs, M)
            if pathOk:
                Paths2Return.append( Path )
        Paths = Paths2Return # Elements of Paths should be all of 0-outage. May contain false positives.
                
        ### Something can be improved here!!!
        if len(Paths) >= 1:
            ## 之後要做這個：
                # 如果在phase 1中 有一些root entry leads to 許多條codeword。那麽記錄下該"問題 root entry"的序號以及其導致的valid paths
                # 先對後面的root entries 進行考察，等到所有root entries都考察完了，再回到這些"問題 root entry"中來，
                    # For all 問題root:
                        # 如果一個"問題root" 現在有且僅有一條在 grand list的path: 那收納該path進 output list. 
                    # 如果一個"問題root" 現在沒有在grand list的path了, 那它的前path 也許是1-outage 或 2-outage，可以用於之後的decoding 
            msg_rt_i = output_message(grand_list, Paths, L, J, messageLens=messageLens)
            decoded_msg = np.vstack((decoded_msg, msg_rt_i)) if decoded_msg.size else msg_rt_i
            # cancel the cdwd sections decoded, or cancel the used root
            pathToCancel = Paths[0]
            
            if SIC:
                for l in range(L):
                    if pathToCancel[l] != -1:
                        grand_list[ pathToCancel[l], l*J:(l+1)*J] = -1*np.ones((J),dtype=int)
            
            # What to do if no SIC???
            # else:
            #     if pChosenRoot == None: 
            #         l = np.argmin(num_erase)
            #         grand_list[ pathToCancel[l], l*J:(l+1)*J] = -1*np.ones((J),dtype=int)
            #     else: 
            #         grand_list[ pathToCancel[pChosenRoot], pChosenRoot*J:(pChosenRoot+1)*J] = -1*np.ones((J),dtype=int)
    decoded_msg = np.unique(decoded_msg, axis=0)
    return decoded_msg, grand_list



def phase2plus_decoder(d, grand_list, L, Gis, columns_index, sub_G_invs, messageLens, parityLens, K, M, SIC=True, pChosenRoots=None):
    
    # Determine the No. of section to perform as the root.
    chosenRoot = 0 if pChosenRoots == None else pChosenRoots[-1]

    # Adjust all corresponding variables according to the newly-chosen root
    erasure_slot = [np.mod(0- chosenRoot, L) for chosenRoot in pChosenRoots] if pChosenRoots!= None else []
    messageLens[range(L)] = messageLens[np.mod(np.arange(chosenRoot, chosenRoot+L),L)]
    parityLens[range(L)] = parityLens[np.mod(np.arange(chosenRoot, chosenRoot+L),L)]
    Gis[range(L)] = Gis[np.mod(np.arange(chosenRoot, chosenRoot+L),L)]
    columns_index[range(L)] = columns_index[np.mod(np.arange(chosenRoot, chosenRoot+L),L)]
    sub_G_invs[range(L)] = sub_G_invs[np.mod(np.arange(chosenRoot, chosenRoot+L),L)]
    grand_list[:, range(L*J)] = grand_list[:, np.mod( np.arange(chosenRoot*J, chosenRoot*J + L*J) ,L*J)]
    Gijs = partition_Gs(L, M, parityLens, Gis) ## Always generate Gijs immediately after Gis.

    K_effective   = [x for x in range(K) if grand_list[x,0] != -1]
    decoded_msg = np.empty(shape=(0,0))


    for i, _ in zip(K_effective, tqdm(range(len(K_effective)))):
        Paths = [ LLC.GLinkedLoop([i], messageLens) ]
        for l in list(range(1,L)): # its last element is L-1
            if len(Paths) == 0: 
                break
            # if d==2: 
            #     print("l="+str(np.mod(chosenRoot + l, L)) +" len=" + str(len(Paths)))
            #     for Path in Paths:
            #         print(Path.get_path(), Path.all_known())
            newAll = []
            survivePaths= Parallel(n_jobs=-1)(delayed(Path_goes_section_l)(l, Paths[j],d, grand_list, K, messageLens, parityLens, 
                                                                           L, M, Gis, Gijs, columns_index, sub_G_invs, erasure_slot) for j in range(len(Paths)))
            # print("This section done.", end=" ")
            for survivePath in survivePaths:
                if len(survivePath) > 0:
                    newAll = list(newAll) + list(survivePath) # list merging
            Paths = newAll 
            # print(l, len(Paths))

        # print(str(i)+"-th root, before final checkng surviving paths=" + str(len(Paths)))
        # if d==2: 
        #     for path in Paths:
        #         # print("***", path.get_path())
        #         print(str( list(np.where( np.array( path.get_path())== -1 )[0]))) 

        PathsUpdated = []
        for j in range(len(Paths)):
            Path = Paths[j]
            isOkay = final_parity_check_oop(Path, grand_list, messageLens, parityLens, L, Gijs, M)
            if isOkay:
                PathsUpdated.append( Path )
        Paths = PathsUpdated
        # print("The root, surviving paths=" + str(len(Paths)))

        if len(Paths) >= 1: # rows inside Paths should be all with one-outage. Some are true positive, some are false positive
            # print(" | We obtained some candidate!!")
            Paths = [Paths[0]]
            # if d==2: print( convert_secNo_to_default(chosenRoot,list(np.where( np.array( Paths[0].get_path())== -1 )[0]) ,L)  ) 
            recovered_message = output_message_oop(grand_list, Paths, L, J)
            decoded_msg = np.vstack((decoded_msg, recovered_message)) if decoded_msg.size else recovered_message
            if SIC:
                pathToCancel = Paths[0].get_path()
                for l in range(L):
                    if pathToCancel[l] != -1:
                        grand_list[ pathToCancel[l], l*J:(l+1)*J] = -1*np.ones((J),dtype=int)
        
    w = sum(messageLens)
    decoded_msg[:,range(w)] = decoded_msg[:, np.mod( np.arange(w) + sum(messageLens[0:L-chosenRoot]), w)]
    decoded_msg = np.unique(decoded_msg, axis=0)
    
    ### shift things back
    grand_list[:, range(L*J)] = grand_list[:, np.mod( np.arange(-chosenRoot*J, -chosenRoot*J + L*J) ,L*J)]
    messageLens[range(L)] = messageLens[np.mod(np.arange(-chosenRoot, -chosenRoot+L),L)]
    parityLens[range(L)] = parityLens[np.mod(np.arange(-chosenRoot, -chosenRoot+L),L)]
    Gis[range(L)] = Gis[np.mod(np.arange(-chosenRoot, -chosenRoot+L),L)]
    columns_index[range(L)] = columns_index[np.mod(np.arange(-chosenRoot, -chosenRoot+L),L)]
    sub_G_invs[range(L)] = sub_G_invs[np.mod(np.arange(-chosenRoot, -chosenRoot+L),L)]

    return decoded_msg, grand_list



def simulation(L, p_e, K, M, channel_type, SIC, txBits, seed):

    messageLens, parityLens = get_allocation(L=L);  N = 2**J # N denotes the length of a codeword, that is rate R = B / N
    ### Retrieve parity-generating matrices from matrix repository
    Gis, columns_index, sub_G_invs = get_G_info(L, M, messageLens, parityLens)
    ### Do partition on Gl's, making them into G_{l,l+1}, G_{l,l+2}, ... , G_{l,l+M}, these matrices with double subscripts are called Gijs
    Gijs = partition_Gs(L, M, parityLens, Gis) 

    ###################################################################################################
    ### Simulation starts.
    print("####### Start Rocking ######## K="+ str(K)+ " and p_e= "+ str(p_e)+ " and L= "+ str(L) +" and M= " + str(M))                                

    ### Encode all messages of K users. Hence tx_cdwds.shape is [K,N]
    tx_cdwds = encode(txBits, K, L, N, M, messageLens, parityLens, Gijs)
    ### Convert binary coded-sub blocks to symbols
    tx_symbols = binary_to_symbol(tx_cdwds, L, K)

    ### B-Channel with Erasure
    rx_symbols, one_outage_where, two_outage_where, n0, n1, n2 = bch_with_erasure(tx_symbols, L, K, p_e, seed=seed)
    if channel_type == "A":
        # A-channel is obtained by removing multiplicities from B-channel
        # We call "rx_symbols" or its equivalence as "the grand list"
        rx_symbols = remove_multiplicity(rx_symbols)

    ### Generate genie reports
    print(" Genie: How many 0-outage? " + str(n0))
    print(" Genie: How many 1-outage? " + str(n1))
    print(" Genie: How many 2-outage? " + str(n2))
    print(" Genie: 1-outage positions: " + str(one_outage_where))
    print(" Genie: 2-outage positions: " + str(two_outage_where))
    ### Convert back to binary representation. (This is what in reality RX can get)
    grand_list = symbol_to_binary(K, L, rx_symbols)
    ###################################################################################################



    ###################################################################################################
    ### Decoding phase 1 (simply finding & stitching 0-outage codewords in the channel output) now starts.
    print(" -- Decoding phase 1 now starts.")
    tic = time.time()
    rxBits_p1, grand_list = phase1_decoder(grand_list, L, Gijs, messageLens, parityLens, K, M, SIC=SIC)
    toc = time.time()
    print(" | Time of phase 1 (LLC): " + str(toc-tic))

    ### If we have >K decoded messages, only choose the first K.
    if rxBits_p1.shape[0] > K: 
        rxBits_p1 = rxBits_p1[np.arange(K)]                    

    ### Check how many are correct amongst the recover (recover means first phase). No need to change.
    all_decoded_txBits = np.unique(rxBits_p1, axis=0)
    txBits_rmd_afterp1 = check_phase(txBits, all_decoded_txBits, "linked loop Code", "1")
    if txBits_rmd_afterp1.shape[0] == B: # Only remains one message 
        txBits_rmd_afterp1 = txBits_rmd_afterp1.reshape(1,-1)
    print(" -Phase 1 Done.\n")
    ###################################################################################################



    ###################################################################################################
    ### Decoding phase 2 (finding/recovering 1-outage codewords in the channel output) now starts.
    print(" -- Decoding phase 2 now starts.")
    tic = time.time()
    rxBits_p21, grand_list= phase2plus_decoder(1, grand_list, L, Gis, columns_index, sub_G_invs, messageLens, parityLens, K, M, SIC=SIC)
    toc = time.time()
    print(" | Time of phase 2.1 " + str(toc-tic))
    txBits_rmd_afterp21 = check_phase(txBits_rmd_afterp1, rxBits_p21, "Linked-loop Code", "2.1")

    tic = time.time()
    rxBits_p22, grand_list= phase2plus_decoder(1, grand_list, L, Gis, columns_index, sub_G_invs, messageLens, parityLens, K, M, SIC=SIC, pChosenRoots=[8])
    toc = time.time()
    print(" | Time of phase 2.2 " + str(toc-tic))
    txBits_rmd_afterp22 = check_phase(txBits_rmd_afterp21, rxBits_p22, "Linked-loop Code", "2.2")

    if rxBits_p21.size: 
        all_decoded_txBits = np.vstack((all_decoded_txBits, rxBits_p21))
    if rxBits_p22.size: 
        all_decoded_txBits = np.vstack((all_decoded_txBits, rxBits_p22))
    all_decoded_txBits = np.unique(all_decoded_txBits, axis=0)
    _ = check_phase(txBits, all_decoded_txBits, "Linked-loop Code", "up-to-phase 2")
    print(" -Phase 2 is done. \n")
    #################################################################################################



    ###################################################################################################
    ### Decoding phase 3 (finding/recovering 2-outage codewords in the channel output) now starts.
    print(" -- Decoding phase 3 now starts.")
    tic = time.time()
    rxBits_p31, grand_list= phase2plus_decoder(2, grand_list, L, Gis, columns_index, sub_G_invs, messageLens, parityLens, K, M, SIC=SIC)
    toc = time.time()
    print(" | Time of phase 3.1 " + str(toc-tic))
    txBits_rmd_afterp31 = check_phase(txBits_rmd_afterp22, rxBits_p31, "Linked-loop Code", "3.1")

    tic = time.time()
    rxBits_p32, grand_list= phase2plus_decoder(2, grand_list, L, Gis, columns_index, sub_G_invs, messageLens, parityLens, K, M, SIC=SIC, pChosenRoots=[6])
    toc = time.time()
    print(" | Time of phase 3.2 " + str(toc-tic))
    txBits_rmd_afterp32 = check_phase(txBits_rmd_afterp31, rxBits_p32, "Linked-loop Code", "3.2")

    tic = time.time()
    rxBits_p33, grand_list= phase2plus_decoder(2, grand_list, L, Gis, columns_index, sub_G_invs, messageLens, parityLens, K, M, SIC=SIC, pChosenRoots=[6,10])
    toc = time.time()
    print(" | Time of phase 3.3 " + str(toc-tic))
    txBits_rmd_afterp33 = check_phase(txBits_rmd_afterp32, rxBits_p33, "Linked-loop Code", "3.3")

    all_decoded_txBits = np.vstack((all_decoded_txBits, rxBits_p31)) if rxBits_p31.size else  all_decoded_txBits
    all_decoded_txBits = np.vstack((all_decoded_txBits, rxBits_p32)) if rxBits_p32.size else  all_decoded_txBits
    all_decoded_txBits = np.vstack((all_decoded_txBits, rxBits_p33)) if rxBits_p33.size else  all_decoded_txBits
    all_decoded_txBits = np.unique(all_decoded_txBits, axis=0)
    _ = check_phase(txBits, all_decoded_txBits, "Linked-loop Code", "up-to-phase 3")
    print(" -Phase 3 is done, this simulation terminates.\n")
    #################################################################################################


    # ###################################################################################################
    # ### Decoding phase 2plus now starts.
    # print(" -- Decoding phase 4 now starts.")
    # tic = time.time()
    # rxBits_p41, grand_list= phase2plus_decoder(3, grand_list, L, Gis, Gijs, columns_index, sub_G_invs, messageLens, parityLens, K, M, SIC=SIC)
    # toc = time.time()
    # print(" | Time of phase 4.1 " + str(toc-tic))
    # txBits_rmd_afterp41 = check_phase(txBits_rmd_afterp33, rxBits_p41, "Linked-loop Code", "4.1")