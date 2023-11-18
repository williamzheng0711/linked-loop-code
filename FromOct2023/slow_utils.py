import numpy as np
import linkedloop as LLC
from utils import *
from binmatrix import *
from slow_lib import *


def llc_decode_compute_parity(Path,cs_decoded_tx_message,J, parityInvolved, toCheck, whichGMatrix, parityLen, messageLen):
    # If path length  = 2, then we wanna have parity for section 2. toCheck = 2
    parityDist = parityInvolved[:,toCheck].reshape(1,-1)[0]
    deciders = np.nonzero(parityDist)[0] # w(decider), where decider \in deciders, decide p(toCheck) collectively
    focusPath = Path[0]

    Parity_computed = np.zeros(parityLen, dtype=int)
    for decider in deciders:      # l labels the sections we gonna check to fix toCheck's parities
        assert whichGMatrix[decider][toCheck] != -1
        gen_mat = matrix_repo(dim=messageLen)[ whichGMatrix[decider][toCheck] ] 
        Parity_computed = Parity_computed + np.matmul( cs_decoded_tx_message[focusPath[decider], decider*J : decider*J+messageLen], gen_mat)
    Parity_computed = np.mod(Parity_computed, 2)
    return Parity_computed


def llc_correct_compute_parity(Path,cs_decoded_tx_message, J, parityInvolved, toCheck, whichGMatrix, parityLen, messageLen):
    # Here, "Path" is a LinkedLoop
    parityDist = parityInvolved[:,toCheck].reshape(1,-1)[0]
    deciders = np.nonzero(parityDist)[0] # w(decider), where decider \in deciders, decide p(toCheck) collectively
    focusPath = Path.get_path()
    # print(toCheck)

    Parity_computed = np.zeros(parityLen, dtype=int)
    for decider in deciders:      # l labels the sections we gonna check to fix toCheck's parities
        useLost = False
        if focusPath[decider] == -1:
            if np.array_equal(Path.get_lostPart(), -1*np.ones((messageLen),dtype=int)):
                return -1 * np.ones((1,parityLen),dtype=int) # We can do nothing here.
            else:
                useLost = True
        assert whichGMatrix[decider][toCheck] != -1
        gen_mat = matrix_repo(dim=messageLen)[ whichGMatrix[decider][toCheck] ] 
        infoInvolved = cs_decoded_tx_message[focusPath[decider], decider*J : decider*J+messageLen] if useLost==False else Path.get_lostPart()
        Parity_computed = Parity_computed + np.matmul( infoInvolved, gen_mat)
    Parity_computed = np.mod(Parity_computed, 2)
    return Parity_computed


def llc_decode_check_parity(Parity_computed,Path,k,cs_decoded_tx_message,J,messageLen):
    Lpath = Path.shape[1]
    Parity = cs_decoded_tx_message[k, Lpath*J+messageLen : (Lpath+1)*J]    # 第k行的第Lpath section的parity
    if (np.sum(np.absolute(Parity_computed-Parity)) == 0):
        return True  
    else: 
        return False 

def llc_correct_lost_by_check_parity(Parity_computed, toCheck, Path, k, cs_decoded_tx_message, J, messageLen,parityLen, parityInvolved, whichGMatrix, L, windowSize):
    # Here, "Path" is a LinkedLoop
    focusPath = Path.get_path()
    oldLostPart = Path.get_lostPart()
    losts = np.where( np.array(focusPath) < 0 )[0]

    generator_matrices = matrix_repo(dim=messageLen)
    inv_generator_matrices = matrix_inv_repo(dim=messageLen)

    # 三種情況不需要檢查：
    # 1. 沒有NA在path中，一切正常。
    # 2. lost 的部分已經被recover出來
    # 3. 
    if Path.whether_contains_na()==False or np.array_equal(oldLostPart,-1*np.ones((messageLen),dtype=int))==False:
        Parity = cs_decoded_tx_message[k,toCheck*J+messageLen:(toCheck+1)*J]    # 第k行的第Lpath section的parity
        if (np.sum(np.absolute(Parity_computed-Parity)) == 0):
            return True, oldLostPart
        else: 
            return False, oldLostPart
    
    # 有lost
    else:   
        lostSection = losts[0]
        # 考慮 lost 發生在最近: # if lostSection is section 4, then we gonna check
        # (w1, w2, w3, w4) => p5    # (w2, w3, w4, w5) => p6    # (w3, w4, w5, w6) => p7    # (w4, w5, w6, w7) => p8    # w5, w6, w7 and w8 are "saverSections" of lostSections
        saverSections = np.nonzero(parityInvolved[lostSection])[0]        # then saverSections = [5, 6, 7, 8]
        availSavers = [saver for saver in saverSections if np.array([np.mod(saver-x,L)<=toCheck for x in range(windowSize+1)]).all() == True ]

        assert len(availSavers) > 0
        if len(availSavers) <= 1:  # Because we need at least TWO results to compare.
            return True, oldLostPart

        theAnswer = np.zeros((messageLen),dtype=int)
        hasAnswer = False

        for saver in availSavers: # saver != lostSection
            assert np.array([saver == np.mod(saver+x,L) for x in range(windowSize+1)]).any() == True
            row = 0
            if saver < toCheck: 
                row = focusPath[saver]
            else: 
                assert saver == toCheck; 
                row = k

            parityDist = parityInvolved[:,saver].reshape(1,-1)[0]
            saverDeciders = np.nonzero(parityDist)[0]
            minuend = cs_decoded_tx_message[row, saver*J+messageLen: (1+saver)*J ]  # 被減數 Aka p(saver)
            subtrahend = np.zeros(parityLen, dtype=int) # 減數
            for saverDecider in saverDeciders:      # l labels the sections we gonna check to fix toCheck's parities
                if (saverDecider != lostSection):
                    gen_mat = generator_matrices[whichGMatrix[saverDecider][saver]] 
                    if saverDecider != toCheck:
                        subtrahend= subtrahend + np.matmul(cs_decoded_tx_message[focusPath[saverDecider],saverDecider*J:saverDecider*J+messageLen],gen_mat) # 都是 info * G
                    else:
                        subtrahend= subtrahend + np.matmul(cs_decoded_tx_message[k,saverDecider*J:saverDecider*J+messageLen],gen_mat) # 都是 info * G

            subtrahend = np.mod(subtrahend, 2)
            gen_mat = generator_matrices[whichGMatrix[lostSection][saver]]
            gen_binmat_inv = np.array(inv_generator_matrices[whichGMatrix[lostSection,saver]])
            theLostPart = np.mod(np.matmul(np.mod(minuend - subtrahend,2),gen_binmat_inv),2)
            if hasAnswer == True:
                # print(theAnswer)
                # print(theLostPart)
                if np.array_equal(theLostPart, theAnswer) == False:
                    return False, oldLostPart
            else:
                theAnswer = theLostPart
                hasAnswer = True
        
        return True, theAnswer



def llc_final_parity_check(Path, cs_decoded_tx_message,J,messageLen,parityLen, parityInvolved, whichGMatrix, L, consider_no_outage = False):
    # Path here is LinkedLoop, must have a lostSection, but not necessarily have it being recovered.
    focusPath = Path.get_path()
    assert len(focusPath) == L
    # if np.array_equal(Path.get_lostPart(), -1*np.ones((messageLen),dtype=int)) == True:
    #     print( "Whrere is lost?" + str(  np.where( np.array(focusPath) < 0)[0]))
    
    # Maybe there are paths that contains no NA. But they must be somewhere not consistent. 
    # if np.count_nonzero(focusPath == -1) == 0:
    if (consider_no_outage == False and focusPath.count(-1) == 0):
        return False

    isOkay = True
    for toCheck in range(L):
        if focusPath[toCheck] != -1:
            parityComputed = llc_correct_compute_parity(Path,cs_decoded_tx_message, J, parityInvolved, toCheck, whichGMatrix, parityLen, messageLen)
            flag_ll = np.abs(parityComputed - cs_decoded_tx_message[focusPath[toCheck], toCheck*J+messageLen: (toCheck+1)*J])
            if flag_ll.any() != 0: 
                isOkay = False
                break
    return isOkay

def output_message(cs_decoded_tx_message, linkedloops, L, J):
    messageLen = linkedloops[0].get_messageLen()
    if len(linkedloops) == 1:
        linkedloop = linkedloops[0]
        path = linkedloop.get_path()
        messageLen = linkedloop.get_messageLen()
        msg = np.zeros( (1, messageLen*L), dtype=int)
        for l in range(L):
            if path[l] != -1: 
                msg[0, l*messageLen:(l+1)*messageLen] = cs_decoded_tx_message[path[l], l*J: l*J+messageLen]
            else: 
                msg[0, l*messageLen:(l+1)*messageLen] = linkedloop.get_lostPart()
        return msg
    elif len(linkedloops) >=2: 
        n = len(linkedloops)
        msg = np.empty( (n, messageLen*L), dtype=int)
        for i in range(n):
            linkedloop = linkedloops[i]
            path = linkedloop.get_path()
            messageLen = linkedloop.get_messageLen()
            for l in range(L):
                if path[l] != -1: 
                    msg[i, l*messageLen:(l+1)*messageLen] = cs_decoded_tx_message[path[l], l*J: l*J+messageLen]
                else: 
                    msg[i, l*messageLen:(l+1)*messageLen] = linkedloop.get_lostPart()
        return msg
            




def slow_decode_deal_with_root_i(i,L,cs_decoded_tx_message, J,parityInvolved, whichGMatrix, messageLen, listSize, parityLen, windowSize):
    # Every i is a root. If section ZERO contains -1, then this root is defective
    if cs_decoded_tx_message[i,0] == -1:
        return -1*np.ones((1,messageLen * L), dtype=int)
    
    Paths = np.array([[i]])
    for l in range(1, L):
        # Grab the parity generator matrix corresponding to this section  
        new=np.empty( shape=(0,0))
        for j in range(Paths.shape[0]):
            Path=Paths[j].reshape(1,-1)     
            Parity_computed= np.ones((1,parityLen),dtype=int)
            if l >= windowSize:
                Parity_computed = llc_decode_compute_parity(Path,cs_decoded_tx_message,J,parityInvolved,l,whichGMatrix,parityLen,messageLen)
            for k in range(listSize):
                if cs_decoded_tx_message[k, l*J] != -1:
                    index = l < windowSize or llc_decode_check_parity(Parity_computed,Path,k,cs_decoded_tx_message,J,messageLen)
                    if index: # If parity constraints are satisfied, update the path
                        new = np.vstack((new,np.hstack((Path.reshape(1,-1),np.array([[k]]))))) if new.size else np.hstack((Path.reshape(1,-1),np.array([[k]])))
        Paths = new 
        if Paths.shape[0] == 0:
            break
    
    # We check those sections at the head, aka, section 0, 1, ..., windowSize-1. We never check them in before.
    # 我們再檢查一下最靠近頭部的 windowSize 個 sectoins，因為在之前的檢查中，它們被跳過了。
    PathsUpdated = np.empty( shape=(0,0))
    for j in range(Paths.shape[0]):
        isOkay = True
        Path = Paths[j].reshape(1,-1)
        for ll in range(windowSize):
            Parity_computed_ll = llc_decode_compute_parity(Path,cs_decoded_tx_message,J, parityInvolved, ll, whichGMatrix, parityLen, messageLen)
            flag_ll = sum( np.abs(Parity_computed_ll - cs_decoded_tx_message[Path[0][ll], ll*J+messageLen: (ll+1)*J]) )
            if flag_ll !=0: 
                isOkay = False; break
        if isOkay:
            PathsUpdated = np.vstack((PathsUpdated, Path)) if PathsUpdated.size else Path
    Paths = PathsUpdated

    # Handle multiple valid paths. We keep all parity consistent paths.
    # 當一個 root 衍生出大於一條合理的 path 時，所有的 path 都會被收錄。
    if Paths.shape[0] >= 1:  
        if Paths.shape[0] >= 2:
            flag = check_if_identical_msgs(Paths, cs_decoded_tx_message, L,J,messageLen)
            if flag:
                return extract_msg_bits(Paths[0].reshape(1,-1),cs_decoded_tx_message, L,J,messageLen)
            else:
                return extract_msg_bits(Paths.reshape(Paths.shape[0],-1),cs_decoded_tx_message, L,J,messageLen)
            # For simplicity, just return ONE path.
        elif Paths.shape[0] == 1:
            return extract_msg_bits(Paths.reshape(1,-1),cs_decoded_tx_message, L,J,messageLen)
    
    # When a root fails to lead to any message, we return an all-minus-one message to indicate failure. 
    # 當一個 root 一條合理的 path 也不能衍生出時，我們輸出一條全是 -1 的固定訊息，以宣告失敗
    return -1*np.ones((1,messageLen * L), dtype=int)





def slow_correct_each_section_and_path(l, Path, cs_decoded_tx_message, J, 
                                       parityInvolved, whichGMatrix, listSize, 
                                       messageLen, parityLen, L, windowSize):
    new = []  
    assert isinstance(Path, LLC.LinkedLoop)
    oldPath = Path.get_path()
    oldLostPart = Path.get_lostPart()
    Parity_computed = -1 * np.ones((1,parityLen),dtype=int)
    
    if l >= windowSize: 
        Parity_computed = llc_correct_compute_parity(Path, cs_decoded_tx_message, J, parityInvolved, l, whichGMatrix, parityLen, messageLen)
    
    for k in range(listSize):
        if cs_decoded_tx_message[k,l*J] != -1:
            if l < windowSize:
                new.append( LLC.LinkedLoop( list(oldPath) + list([k]) , messageLen, oldLostPart) )
            else : 
                toKeep, lostPart = llc_correct_lost_by_check_parity(Parity_computed, l, Path, k, cs_decoded_tx_message, J, messageLen,parityLen,parityInvolved, whichGMatrix, L, windowSize)
                if toKeep:
                    new.append( LLC.LinkedLoop( list(oldPath) + list([k]) , messageLen, lostPart) )
    
    if Path.whether_contains_na() == False:
        if l != L-1:
            new.append( LLC.LinkedLoop( list(oldPath) + list([-1]) , messageLen, oldLostPart) )
        else:
            savers = list(range(windowSize)) # E.g. [0,1], we gonna use w(L-1)G0 +    w(0)G1    = p(1)
            parity_section = windowSize-1
            parity_save = cs_decoded_tx_message[oldPath[parity_section], parity_section*J+messageLen : (parity_section+1)*J]
            partial_sum = np.zeros((messageLen), dtype=int)
            for saver in savers:
                gen_mat = matrix_repo(dim=messageLen)[ whichGMatrix[saver][parity_section] ]
                partial_sum = partial_sum + np.matmul(cs_decoded_tx_message[oldPath[saver],saver*J:saver*J+messageLen],gen_mat)
            infoG = np.mod(parity_save - partial_sum, 2)
            G_inv = matrix_inv_repo(dim=messageLen)[ whichGMatrix[L-1][parity_section] ]
            info = np.mod( np.matmul(infoG, G_inv), 2)
            new.append( LLC.LinkedLoop( list(oldPath) + list([-1]) , messageLen, info) ) 
    return new