import numpy as np
import linkedloop as LLC
import binmatrix as BM

from static_repo import *

def who_decides_p_sec(L, l, M):
    return [(l-j) % L for j in range(M,0,-1)]

# For phase 1
def compute_parity(L, Path, rx_cdwds, toCheck, messageLens, parityLens, Gijs, M):
    deciders = who_decides_p_sec(L,toCheck,M)
    Parity_computed = np.zeros(parityLens[toCheck], dtype=int)
    for decider in deciders:      # l labels the sections we gonna check to fix toCheck's parities
        gen_mat = Gijs[ CantorPairing(decider, toCheck) ]
        infoInvolved = rx_cdwds[Path[decider], decider*J : decider*J+messageLens[decider]]
        Parity_computed = Parity_computed + np.matmul( infoInvolved, gen_mat)
    Parity_computed = np.mod(Parity_computed, 2)
    return Parity_computed

# For phase 1
def final_parity_check(Path, rx_cdwds, messageLens, parityLens, L, Gijs, M):
    assert len(Path)== L
    decision= True
    for m in range(M):
        parityComputed = compute_parity(L, Path, rx_cdwds, m, messageLens, parityLens, Gijs, M)
        should_be_0 = np.abs(parityComputed - rx_cdwds[Path[m], m*J+messageLens[m]: (m+1)*J])
        if should_be_0.any() != 0: 
            decision = False
            break
    return decision

# For phase 1
def output_message(grand_list, paths, L, J, messageLens):
    n = len(paths)
    msg = np.empty( (n, B), dtype=int)
    for i in range(n):
        path = paths[i]
        for l in range(L):
            msg[i, sum(messageLens[0:l]): sum(messageLens[0:l])+messageLens[l]] = grand_list[path[l], l*J: l*J+messageLens[l]]
    return msg


# For phase 2
def compute_parity_oop(L, Path, grand_list, toCheck, messageLens, parityLens, Gijs, M):
    # Here, "Path" is a LinkedLoop
    focusPath = Path.get_path()
    deciders = who_decides_p_sec(L, toCheck, M)

    Parity_computed = np.zeros(parityLens[toCheck], dtype=int)
    for decider in deciders:
        useLost = False
        if focusPath[decider] == -1:
            if np.array_equal(Path.get_lostPart(), np.empty((0),dtype=int)): 
                return -1 * np.ones((1,parityLens[toCheck]),dtype=int)          # We can do nothing here.
            else:
                useLost = True
        assert CantorPairing(decider, toCheck)!= -1
        gen_mat = Gijs[ CantorPairing(decider, toCheck) ]
        infoInvolved = grand_list[focusPath[decider], decider*J : decider*J+messageLens[decider]] if useLost==False else Path.get_lostPart()
        Parity_computed = Parity_computed + np.matmul( infoInvolved, gen_mat)
    
    Parity_computed = np.mod(Parity_computed, 2)
    return Parity_computed


# For phase 2
def GLLC_grow_a_consistent_path(Parity_computed, toCheck, Path, k, grand_list, messageLens, parityLens,
                                L, M, Gis, Gijs, columns_index, sub_G_invs):
    # Here, "Path" is a LinkedLoop
    focusPath = Path.get_path()
    oldLostPart = Path.get_lostPart()
    losts = np.where( np.array(focusPath) < 0 )[0]


    # Under the following circumstances, we don't need to consider about RECOVERING something lost
    # 1. There is nothing lost in the sub-path at all. Aka, no "na". and 
    # 2. The lost part has already been recovered long time ago. (> windowSize)
    if Path.whether_contains_na()==False or np.array_equal(oldLostPart, np.empty((0),dtype=int))==False:
        Parity = grand_list[k, toCheck*J+messageLens[toCheck] :(toCheck+1)*J ]    # 第k行的第 toCheck section的parity
        if (np.sum(np.absolute(Parity_computed-Parity)) == 0):
            return True, oldLostPart
        else: 
            return False, oldLostPart
    
    # 有lost
    else:   
        lostSection = losts[0]
        # (w1, w2, w3, w4) => p5    # (w2, w3, w4, w5) => p6    
        # (w3, w4, w5, w6) => p7    # (w4, w5, w6, w7) => p8    
        # # w5, w6, w7 and w8 are "saverSections" of lostSections. w(4) is what we lost.
        saverSections = [ np.mod(lostSection+ll, L) for ll in range(1,M+1,1) ]        # then saverSections = [5, 6, 7, 8]
        availSavers = [saver for saver in saverSections if np.array([np.mod(saver-x,L)<=toCheck for x in range(M+1)]).all() == True]

        assert len(availSavers) > 0
        if sum(parityLens[availSavers]) < messageLens[lostSection]:
            return True, oldLostPart

        theAnswer = np.zeros((messageLens[lostSection]),dtype=int)        
        hasAnswer = (np.array_equal(oldLostPart, np.empty((0),dtype=int))==False)    # 如果沒有存答案，那就是 hasAnswer = False
        known_vectors = []
    

        longMinued = np.empty((0),dtype=int)
        longSubtrahend = np.empty((0),dtype=int)
        for availSaver in availSavers: # saver != lostSection            
            saverDeciders = [ np.mod(availSaver-ll, L) for ll in range(M,0,-1) ] # 被減數 Aka p(saver)
            minuend = grand_list[focusPath[availSaver], availSaver*J+messageLens[availSaver]: (availSaver+1)*J ] if Path.known_in_path(availSaver) else grand_list[k, availSaver*J+messageLens[availSaver]: (availSaver+1)*J] 
            longMinued = np.hstack( (longMinued, np.mod(minuend,2)))
            subtrahend = np.zeros((parityLens[availSaver]),dtype=int)
    
            for saverDecider in saverDeciders:     
                if (saverDecider != lostSection):
                    gen_mat = Gijs[ CantorPairing(saverDecider, availSaver) ]
                    toAdd = np.matmul(grand_list[focusPath[saverDecider],saverDecider*J:saverDecider*J+messageLens[saverDecider]], gen_mat) if saverDecider != toCheck else np.matmul(grand_list[k,saverDecider*J:saverDecider*J+messageLens[saverDecider]], gen_mat)
                    toAdd = np.mod(toAdd, 2)
                    subtrahend = subtrahend + toAdd

            subtrahend = np.mod(subtrahend, 2)
            longSubtrahend = np.hstack((longSubtrahend, np.mod(subtrahend,2)) )
            known_vectors.append(np.mod(minuend - subtrahend,2))
        
        concatenated_known_vctr = np.empty((0),dtype=int)
        for known_vector in known_vectors:
            concatenated_known_vctr = np.hstack((concatenated_known_vctr, known_vector))

        # !!!! This is a temporary patch, should be dealt later when considering more general codes
        if lostSection >= M - 1:
            sufficent_columns = np.array(columns_index[lostSection],dtype=int)
            gen_binmat_inv = sub_G_invs[lostSection]
            theLostPart = np.mod(np.matmul(concatenated_known_vctr[sufficent_columns],gen_binmat_inv),2)
            assert theLostPart.shape[0] == messageLens[lostSection]
        else: 
            sufficent_columns = range(8, 16)
            BinMat = BM.BinMatrix(m= Gis[lostSection][:,sufficent_columns])
            gen_binmat_inv = np.array(BinMat.inv(), dtype=int )
            theLostPart = np.mod(np.matmul(concatenated_known_vctr[range(8)],gen_binmat_inv),2)
            assert theLostPart.shape[0] == messageLens[lostSection]

        if np.array_equal(np.mod( np.matmul(theLostPart, Gis[lostSection]), 2), concatenated_known_vctr) == False and len(availSavers)==M:
            return False , oldLostPart

        if hasAnswer == True:
            if np.array_equal(theLostPart, theAnswer) == False:
                return False, oldLostPart
        else:
            theAnswer = theLostPart
            hasAnswer = True

        return True, theLostPart




# For phase 2
def GLLC_correct_each_section_and_path(sec2chk, Path, grand_list, K, messageLens, 
                                       parityLens, L, M, Gis, Gijs, columns_index, sub_G_invs, num_erase):
    new = []  
    assert isinstance(Path, LLC.GLinkedLoop)
    oldPath = Path.get_path()
    oldLostPart = Path.get_lostPart()
    oldLostSection = Path.get_lostSection()
    Parity_computed = np.empty((0),dtype=int)

    if sec2chk >= M: 
        Parity_computed = compute_parity_oop(L, Path, grand_list, sec2chk, messageLens, parityLens, Gijs, M)
    
    for k in range(K):
        if grand_list[k,sec2chk*J] != -1:
            if sec2chk < M: # the sub-path on hand is too short, hence is impossible to be inconsistent
                new.append( LLC.GLinkedLoop( list(oldPath) + list([k]) , messageLens, oldLostPart, oldLostSection) )
            else : 
                toKeep, lostPart = GLLC_grow_a_consistent_path(Parity_computed=Parity_computed, toCheck=sec2chk, Path=Path, k=k, 
                                                                         cs_decoded_tx_message=grand_list, J=J, messageLens=messageLens,
                                                                         parityLens= parityLens, whichGMatrix=whichGMatrix, L=L,
                                                                         windowSize=M, Gs=Gs, Gijs=Gijs,
                                                                         columns_index= columns_index, sub_G_inversions=sub_G_inversions)
                if toKeep:
                    # print(lostPart)
                    new.append( LLC.GLinkedLoop( list(oldPath) + list([k]) , messageLens, lostPart, oldLostSection) )
    
    if Path.whether_contains_na() == False and num_erase[sec2chk]!=0:
        if sec2chk != L-1:
            new.append( LLC.GLinkedLoop( list(oldPath) + list([-1]), messageLens, oldLostPart, sec2chk) )
        else:
            temp_path = LLC.GLinkedLoop( list(oldPath) + list([-1]), messageLens, oldLostPart)
            toKeep, lostPart = GLLC_grow_a_consistent_path(Parity_computed = None, toCheck=L-1, Path=temp_path, k= None, 
                                                                    cs_decoded_tx_message=grand_list, J=J, messageLens=messageLens, parityLens=parityLens,
                                                                    whichGMatrix=whichGMatrix, L=L, windowSize=windowSize, Gs=Gs, Gijs=Gijs,
                                                                    columns_index=columns_index, sub_G_inversions=sub_G_inversions)
            if toKeep:
                new.append( LLC.GLinkedLoop( list(oldPath) + list([-1]) , messageLens, lostPart, sec2chk) ) 
    
    return new


# Done
def GLLC_final_parity_check(Path, cs_decoded_tx_message,J,messageLens, parityLens, whichGMatrix, L, Gijs, windowSize):
    focusPath = Path.get_path()
    assert len(focusPath) == L
    isOkay = True
    markdown = False
    # print(Path.get_lostSection())
    # if Path.get_lostSection() == 1:
    #     markdown = True
    #     print("Focus !!!")
    #     print( np.array(Path.get_lostPart(), dtype=int) )

    for toCheck in range(L):
        if focusPath[toCheck] != -1:
            parityComputed = GLLC_compute_parity(L=L, Path=Path,cs_decoded_tx_message=cs_decoded_tx_message, 
                                J=J, toCheck=toCheck, whichGMatrix=whichGMatrix, messageLens=messageLens, 
                                parityLens=parityLens, Gijs=Gijs, windowSize=windowSize)
            flag_ll = np.abs(parityComputed - cs_decoded_tx_message[focusPath[toCheck], toCheck*J+messageLens[toCheck]: (toCheck+1)*J])
            if flag_ll.any() != 0: 
                isOkay = False
                break
    
    if markdown: 
        print(isOkay)

    return isOkay

    

# Done
def GLLC_output_message(cs_decoded_tx_message, linkedloops, L, J):
    messageLens = linkedloops[0].get_messageLens()
    n = len(linkedloops)
    msg = np.empty( (n, sum(messageLens)), dtype=int)
    for i in range(n):
        linkedloop = linkedloops[i]
        path = linkedloop.get_path()
        for l in range(L):
            if path[l] != -1: 
                msg[i, sum(messageLens[0:l]): sum(messageLens[0:l])+messageLens[l]] = cs_decoded_tx_message[path[l], l*J: l*J+messageLens[l]]
            else: 
                msg[i, sum(messageLens[0:l]): sum(messageLens[0:l])+messageLens[l]] = linkedloop.get_lostPart()
    return msg



def check_phase(txBits, rxBits_phase, name, phase):
    # Check how many are correct amongst the recover (recover means first phase). No need to change.
    if len(rxBits_phase) == 0:
        return txBits
    
    thisIter = 0
    txBits_remained = np.empty(shape=(0,0))
    for i in range(txBits.shape[0]):
        incre = 0
        incre = np.equal(txBits[i,:],rxBits_phase).all(axis=1).any()
        thisIter += int(incre)
        if (incre == False):
            txBits_remained = np.vstack( (txBits_remained, txBits[i,:]) ) if txBits_remained.size else  txBits[i,:]
    print(" | In phase " + phase + " " + str(name) + " decodes " + str(thisIter) + " true message out of " +str(rxBits_phase.shape[0]))
    return txBits_remained