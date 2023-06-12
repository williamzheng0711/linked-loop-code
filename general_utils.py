import numpy as np
import linkedloop as LLC


def GLLC_encode(tx_message, K, L, J, Pa, w, messageLens, parityLens, Gs, windowSize, Gijs):
    """
    Parameters
    ----------
    tx_message (ndarray): K x w matrix of K users' w-bit messages
    K (int): number of active users
    L (int): number of sections in codeword
    J (int): number of bits/section

    Returns
    -------
    encoded_tx_message : ndarray (K by (w+Pa) matrix, or 100 by 256 in usual case)
    """
    encoded_tx_message = np.zeros((K, w+Pa), dtype=int)
    for l in range(L):
        encoded_tx_message[:,l*J:l*J+messageLens[l] ] = tx_message[: , sum(messageLens[0:l]) :sum(messageLens[0:l]) + messageLens[l]]
        who_decides_pl = [(l-j) % L for j in range(windowSize,0,-1)]
        parity_l = np.zeros((K, parityLens[l]), dtype=int)
        for decider in who_decides_pl: 
            # print( Gijs[2**decider*3**l].shape, decider, l)
            toAdd = (tx_message[:,sum(messageLens[0:decider]):sum(messageLens[0:decider])+ messageLens[decider]] @ Gijs[2**decider*3**l] )
            parity_l = parity_l + toAdd
        encoded_tx_message[: , l*J+messageLens[l]:(l+1)*J] = np.mod(parity_l, 2)

    # One can check what a outer-encoded message looks like in the csv file.
    # np.savetxt('encoded_message.csv', encoded_tx_message[0].reshape(16,16), delimiter=',', fmt='%d')

    return encoded_tx_message


# Done
def GLLC_correct_each_section_and_path(section2Check, Path, cs_decoded_tx_message, J, whichGMatrix, K, messageLens, 
                                       parityLens, L, windowSize, Gs, Gijs, columns_index, sub_G_inversions, num_erase):
    # print(Path.get_path())
    new = []  
    assert isinstance(Path, LLC.GLinkedLoop)
    oldPath = Path.get_path()
    oldLostPart = Path.get_lostPart()
    Parity_computed = np.empty((0),dtype=int)

    if section2Check >= windowSize: 
        Parity_computed = GLLC_compute_parity(L=L, Path=Path, cs_decoded_tx_message=cs_decoded_tx_message, 
                                                          J=J, toCheck=section2Check, whichGMatrix= whichGMatrix, messageLens=messageLens, 
                                                          parityLens=parityLens, Gijs=Gijs, windowSize=windowSize)
    for k in range(K):
        if cs_decoded_tx_message[k,section2Check*J] != -1:
            if section2Check < windowSize:
                new.append( LLC.GLinkedLoop( list(oldPath) + list([k]) , messageLens, oldLostPart) )
            else : 
                toKeep, lostPart = GLLC_grow_a_consistent_path(Parity_computed=Parity_computed, toCheck=section2Check, Path=Path, k=k, 
                                                                         cs_decoded_tx_message=cs_decoded_tx_message, J=J, messageLens=messageLens,
                                                                         parityLens= parityLens, whichGMatrix=whichGMatrix, L=L,
                                                                         windowSize=windowSize, Gs=Gs, Gijs=Gijs,
                                                                         columns_index= columns_index, sub_G_inversions=sub_G_inversions)
                if toKeep:
                    # print(lostPart)
                    new.append( LLC.GLinkedLoop( list(oldPath) + list([k]) , messageLens, lostPart) )
    
    if Path.whether_contains_na() == False and num_erase[section2Check]!=0 :
        if section2Check != L-1:
            new.append( LLC.GLinkedLoop( list(oldPath) + list([-1]), messageLens, oldLostPart) )
        else:
            temp_path = LLC.GLinkedLoop( list(oldPath) + list([-1]), messageLens, oldLostPart)
            toKeep, lostPart = GLLC_grow_a_consistent_path(Parity_computed = None, toCheck=L-1, Path=temp_path, k= None, 
                                                                    cs_decoded_tx_message=cs_decoded_tx_message, J=J, messageLens=messageLens, parityLens=parityLens,
                                                                    whichGMatrix=whichGMatrix, L=L, windowSize=windowSize, Gs=Gs, Gijs=Gijs,
                                                                    columns_index=columns_index, sub_G_inversions=sub_G_inversions)
            if toKeep:
                new.append( LLC.GLinkedLoop( list(oldPath) + list([-1]) , messageLens, lostPart) ) 
    return new


# Done
def GLLC_final_parity_check(Path, cs_decoded_tx_message,J,messageLens, parityLens, whichGMatrix, L, Gijs, windowSize):
    focusPath = Path.get_path()
    assert len(focusPath) == L
    isOkay = True
    for toCheck in range(L):
        if focusPath[toCheck] != -1:
            parityComputed = GLLC_compute_parity(L=L, Path=Path,cs_decoded_tx_message=cs_decoded_tx_message, 
                                J=J, toCheck=toCheck, whichGMatrix=whichGMatrix, messageLens=messageLens, 
                                parityLens=parityLens, Gijs=Gijs, windowSize=windowSize)
            flag_ll = np.abs(parityComputed - cs_decoded_tx_message[focusPath[toCheck], toCheck*J+messageLens[toCheck]: (toCheck+1)*J])
            if flag_ll.any() != 0: 
                isOkay = False
                break
    return isOkay


# Done
def GLLC_compute_parity(L, Path,cs_decoded_tx_message, J, toCheck, whichGMatrix, messageLens, parityLens, Gijs, windowSize):
    # Here, "Path" is a LinkedLoop
    focusPath = Path.get_path()
    deciders = np.mod([ toCheck + j for j in range(-1*windowSize, 0, 1) ], L)

    Parity_computed = np.zeros(parityLens[toCheck], dtype=int)
    for decider in deciders:      # l labels the sections we gonna check to fix toCheck's parities
        useLost = False
        if focusPath[decider] == -1:
            if np.array_equal(Path.get_lostPart(), np.empty((0),dtype=int)):
                return -1 * np.ones((1,parityLens[toCheck]),dtype=int) # We can do nothing here.
            else:
                useLost = True
        # 能走到這裡的已經是 lostPart不為空的了
        assert whichGMatrix[decider][toCheck] != -1
        gen_mat = Gijs[ whichGMatrix[decider, toCheck] ]
        infoInvolved = cs_decoded_tx_message[focusPath[decider], decider*J : decider*J+messageLens[decider]] if useLost==False else Path.get_lostPart()
        Parity_computed = Parity_computed + np.matmul( infoInvolved, gen_mat)
    Parity_computed = np.mod(Parity_computed, 2)

    return Parity_computed


# Done
def GLLC_grow_a_consistent_path(Parity_computed, toCheck, Path, k, cs_decoded_tx_message, J, messageLens, parityLens,
                                whichGMatrix, L, windowSize, Gs, Gijs, columns_index, sub_G_inversions):
    # Here, "Path" is a LinkedLoop
    focusPath = Path.get_path()
    oldLostPart = Path.get_lostPart()
    losts = np.where( np.array(focusPath) < 0 )[0]

    # 下列情況不需要檢查：
    # 1. 沒有NA在path中，一切正常。  2. lost 的部分已經被recover出來
    if Path.whether_contains_na()==False or np.array_equal(oldLostPart, np.empty((0),dtype=int))==False:
        Parity = cs_decoded_tx_message[k, toCheck*J+messageLens[toCheck] :(toCheck+1)*J ]    # 第k行的第 toCheck section的parity
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
        saverSections = [ np.mod(lostSection+ll, L) for ll in range(1,windowSize+1,1) ]        # then saverSections = [5, 6, 7, 8]
        availSavers = [saver for saver in saverSections if np.array([np.mod(saver-x,L)<=toCheck for x in range(windowSize+1)]).all() == True]

        assert len(availSavers) > 0
        if len(availSavers) < windowSize:  # Because we need at least TWO results to compare.
            return True, oldLostPart

        theAnswer = np.zeros((messageLens[lostSection]),dtype=int)        
        hasAnswer = (np.array_equal(oldLostPart, np.empty((0),dtype=int))==False)    # 如果沒有存答案，那就是 hasAnswer = False
        known_vectors = []
    

        longMinued = np.empty((0),dtype=int)
        longSubtrahend = np.empty((0),dtype=int)
        for availSaver in availSavers: # saver != lostSection            
            # If saver=3, windowSize=2. saverDeciders = [1,2]
            saverDeciders = [ np.mod(availSaver-ll, L) for ll in range(windowSize,0,-1) ] # 被減數 Aka p(saver)
            minuend = cs_decoded_tx_message[focusPath[availSaver], availSaver*J+messageLens[availSaver]: (availSaver+1)*J ] if Path.known_in_path(availSaver) else cs_decoded_tx_message[k, availSaver*J+messageLens[availSaver]: (availSaver+1)*J] 
            longMinued = np.hstack( (longMinued, np.mod(minuend,2)))
            # print("Minued " + str(known_vec1_comp1) +" "+ str(known_vec2_comp1) +" ||||| "+ str(minuend))
            subtrahend = np.zeros((parityLens[availSaver]),dtype=int) # 減數    被減數和減數 的size當然是一樣的
            # subtrahend 永遠都是 info * G 這樣的格式
            # print(known_vec1_comp2, known_vec2_comp2)
    
            for saverDecider in saverDeciders:     
                if (saverDecider != lostSection):
                    gen_mat = Gijs[ whichGMatrix[saverDecider, availSaver] ]
                    # gen_mat = Gijs[2**saverDecider*3**avialSaver]
                    toAdd = np.matmul(cs_decoded_tx_message[focusPath[saverDecider],saverDecider*J:saverDecider*J+messageLens[saverDecider]], gen_mat) if saverDecider != toCheck else np.matmul(cs_decoded_tx_message[k,saverDecider*J:saverDecider*J+messageLens[saverDecider]], gen_mat)
                    toAdd = np.mod(toAdd, 2)
                    subtrahend = subtrahend + toAdd

            subtrahend = np.mod(subtrahend, 2)
            longSubtrahend = np.hstack((longSubtrahend, np.mod(subtrahend,2)) )
            known_vectors.append(np.mod(minuend - subtrahend,2))

        # print("被減數 " + str(longMinued), known_vec1_comp1, known_vec2_comp1)
        # print("減數 " + str(longSubtrahend), known_vec1_comp2, known_vec2_comp2)
        # print("infoG: " +str(known_vec1_comp2) + " " + str(known_vec2_comp2) +" *** "+ str(box))
        # print("difference? " + str(known_vec) +"  " + str(known_vectors))
        
        concatenated_known_vctr = np.empty((0),dtype=int)
        for known_vector in known_vectors:
            concatenated_known_vctr = np.hstack((concatenated_known_vctr, known_vector))

        # print(known_vec1,known_vectors[0], known_vec2, known_vectors[1])
        # print("***" + str(known_vec) +"  "  +str(concatenated_known_vctr))

        sufficent_columns = columns_index[lostSection]
        gen_binmat_inv = sub_G_inversions[lostSection]

        theLostPart = np.mod(np.matmul(concatenated_known_vctr[sufficent_columns],gen_binmat_inv),2)
        assert theLostPart.shape[0] == messageLens[lostSection]

        if np.array_equal(np.mod( np.matmul(theLostPart, Gs[lostSection]), 2), concatenated_known_vctr) == False:
            return False , oldLostPart

        if hasAnswer == True:
            if np.array_equal(theLostPart, theAnswer) == False:
                return False, oldLostPart
        else:
            theAnswer = theLostPart
            hasAnswer = True

        return True, theLostPart
    

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
    # print(msg.shape)
    return msg