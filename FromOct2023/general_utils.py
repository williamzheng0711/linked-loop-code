import numpy as np
import linkedloop as LLC
import binmatrix as BM

from static_repo import *


def convert_secNo_to_default(chosenRoot, sect, L):
    return np.mod( [chosenRoot + a for a in sect], L)

def convert_sec_to_default(chosenRoot, x, L):
    return np.mod( chosenRoot + x, L)

def who_decides_p_sec(L, l, M):
    return [(l-j) % L for j in range(M,0, -1)]

def I_decides_who(L, l, M):
    return [(l+j) % L for j in range(1,M+1,1)]  

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

    # if toCheck == 2 and focusPath[1]==-1:   assert 1 not in Path.get_dictLostInfos()

    Parity_computed = np.zeros(parityLens[toCheck], dtype=int)
    for decider in deciders:
        useLost = False
        if focusPath[decider] == -1:
            if decider not in Path.get_dictLostInfos(): 
                return -2 * np.ones(parityLens[toCheck],dtype=int)          # We can do nothing here.
            else:
                useLost = True
        gen_mat = Gijs[ CantorPairing( decider, toCheck) ]
        # if useLost == True: print(Path.get_dictLostInfos(), decider)
        infoInvolved = grand_list[focusPath[decider], decider*J : decider*J+messageLens[decider]] if useLost==False else Path.get_dictLostInfos()[decider]
        Parity_computed = Parity_computed + np.matmul( infoInvolved, gen_mat)
    
    Parity_computed = np.mod(Parity_computed, 2)
    return Parity_computed
    # return -1 * np.ones((1,parityLens[toCheck]),dtype=int)



# For phase 2+
def Path_goes_entry_k(d, Parity_computed, toCheck, Path, k, grand_list, messageLens, parityLens,
                                L, M, Gis, Gijs, columns_index, sub_G_invs):
    # Here, "Path" is a LinkedLoop
    focusPath = Path.get_path()
    oldDictLostInfos = Path.get_dictLostInfos().copy()
    losts = np.where( np.array(focusPath) < 0 )[0]

    # Under the following circumstances, we don't need to consider about RECOVERING something lost
    # 1. There is nothing lost in the sub-path at all. Aka, no "na". or 
    # 2. The lost part has already been recovered. 
    toCheckDeciders = who_decides_p_sec(L, toCheck, M)
    canDecide_toCheck = True
    for toCheckDecider in toCheckDeciders:
        if (focusPath[toCheckDecider]==-1) and (toCheckDecider not in oldDictLostInfos):
            canDecide_toCheck = False

    # if d==2 and len(losts)==2 and consecutive(losts): print(Parity_computed, focusPath, toCheck, Path.num_na_in_path(), Path.get_dictLostInfos(), Path.get_listLostSects())

    if canDecide_toCheck and toCheck != L-1 :
        Parity = grand_list[k, toCheck*J+messageLens[toCheck] :(toCheck+1)*J ]    # 第k行的第 toCheck section的parity
        if (np.sum(np.absolute(Parity_computed-Parity)) == 0):
            return True, oldDictLostInfos
        else: 
            return False, {}

    # 有lost
    else:   
        unsolved_losts = [x for x in losts if x not in oldDictLostInfos]
        for lostSection in unsolved_losts:
            # # w5, w6, w7 and w8 are "saverSections" of lostSections. w(4) is what we lost.
            saverSections = I_decides_who(L, lostSection, M)             # then saverSections = [5, 6, 7, 8]
            availSavers = [saver for saver in saverSections if np.array([np.mod(saver-x,L)<=toCheck for x in range(M+1)]).all() == True]
            # if len(focusPath)<=2 : print(saverSections, availSavers, focusPath, Path.get_dictLostInfos())

            assert len(availSavers) > 0
            if sum(parityLens[availSavers]) < messageLens[lostSection]: 
                return True, oldDictLostInfos ## 存疑

            known_vectors = []
        
            longMinued = np.empty((0),dtype=int)
            longSubtrahend = np.empty((0),dtype=int)
            for availSaver in availSavers: # saver != lostSection            
                saverDeciders = who_decides_p_sec(L, availSaver, M)   # 被減數 Aka p(saver)
                minuend = grand_list[focusPath[availSaver], availSaver*J+messageLens[availSaver]: (availSaver+1)*J ] if Path.known_in_path(availSaver) else grand_list[k, availSaver*J+messageLens[availSaver]: (availSaver+1)*J] 
                longMinued = np.hstack( (longMinued, np.mod(minuend,2)))
                subtrahend = np.zeros((parityLens[availSaver]),dtype=int)
        
                for saverDecider in saverDeciders:     
                    if (saverDecider != lostSection):
                        gen_mat = Gijs[ CantorPairing(saverDecider, availSaver) ]
                        try:
                            toAdd = np.matmul(grand_list[focusPath[saverDecider],saverDecider*J:saverDecider*J+messageLens[saverDecider]], gen_mat) if saverDecider != toCheck else np.matmul(grand_list[k,saverDecider*J:saverDecider*J+messageLens[saverDecider]], gen_mat)
                        except:
                            print(str(saverDecider != toCheck) + " " + str(k) + " " + str(losts))
                            if saverDecider != toCheck:
                                print(grand_list[focusPath[saverDecider],saverDecider*J:saverDecider*J+messageLens[saverDecider]])
                            else:
                                print(grand_list[k,saverDecider*J:saverDecider*J+messageLens[saverDecider]])
                        # except:
                        #     # do something rubbish
                        #     toAdd = np.matmul(grand_list[focusPath[0], saverDecider*J:saverDecider*J+messageLens[saverDecider]], gen_mat)
                        toAdd = np.mod(toAdd, 2)
                        subtrahend = subtrahend + toAdd

                subtrahend = np.mod(subtrahend, 2)
                longSubtrahend = np.hstack((longSubtrahend, np.mod(subtrahend,2)) )
                known_vectors.append(np.mod(minuend - subtrahend,2))
            
            concatenated_known_vctr = np.empty((0),dtype=int)
            for known_vector in known_vectors:
                concatenated_known_vctr = np.hstack((concatenated_known_vctr, known_vector))

            # !!!! This is a temporary patch, should be dealt later when considering more general codes
            ## This only works for (L=16, M=3) cases
            # if lostSection >= M - 1:

            ## This means that erasures happening at [0,1] will not be dealt
            sufficent_columns = np.array(columns_index[lostSection],dtype=int)
            gen_binmat_inv = sub_G_invs[lostSection]
            newAnswer = np.array(np.mod(np.matmul(concatenated_known_vctr[sufficent_columns],gen_binmat_inv),2), dtype=int)
            assert newAnswer.shape[0] == messageLens[lostSection]
            # else: 
            #     assert lostSection == 1
            #     # if d==1: print("来了", str(availSavers))
            #     sufficent_columns = range(8, 16)   # which is not the usual range(0,8)
            #     BinMat = BM.BinMatrix(m= Gis[lostSection][:,sufficent_columns])
            #     gen_binmat_inv = np.array(BinMat.inv(), dtype=int)
            #     newAnswer = np.array( np.mod(np.matmul(concatenated_known_vctr[range(8)],gen_binmat_inv),2), dtype= int)
            #     assert newAnswer.shape[0] == messageLens[lostSection]

            if np.array_equal(np.mod( np.matmul(newAnswer, Gis[lostSection]), 2), concatenated_known_vctr) == False and len(availSavers)==M:
                return False , {}
            oldDictLostInfos[lostSection] = newAnswer
            # if d==2: print("有作用")
        return True, oldDictLostInfos




# For phase 2
def Path_goes_section_l(l, Path, d, grand_list, K, messageLens, parityLens, L, M, Gis, Gijs, columns_index, sub_G_invs, erasure_slot):
    new = []  
    assert isinstance(Path, LLC.GLinkedLoop)
    oldPath = Path.get_path().copy()
    oldListLostSects = Path.get_listLostSects().copy()
    oldDictLostInfos = Path.get_dictLostInfos().copy()
    Parity_computed = np.empty((0),dtype=int)

    if l >= M: 
        Parity_computed = compute_parity_oop(L, Path, grand_list, l, messageLens, parityLens, Gijs, M)
        # if sum(Parity_computed) < 0 : print("AAAA", Path.get_path(), Path.get_listLostSects(), Path.get_dictLostInfos()) 
    for k in range(K):
        if grand_list[k,l*J] != -1:
            ########### Problems are here
            if l < M : # the sub-path on hand is too short, hence is impossible to be inconsistent
                new.append( LLC.GLinkedLoop( list(oldPath) + list([k]), messageLens, oldListLostSects, oldDictLostInfos) )
            else : 
                Path = LLC.GLinkedLoop(oldPath, messageLens, oldListLostSects, oldDictLostInfos )
                toKeep, updDictLostInfos = Path_goes_entry_k(d, Parity_computed, l, Path, k, grand_list, messageLens, parityLens, L, M, Gis, Gijs, columns_index, sub_G_invs)
                if toKeep:
                    new.append(LLC.GLinkedLoop(list(oldPath)+ list([k]), messageLens, oldListLostSects, updDictLostInfos))
    
    # if Path.num_na_in_path() < d and (erasure_slot== None   or   l in erasure_slot    or   subset(erasure_slot, oldListLostSects)):
    if d-Path.num_na_in_path()>0 and ( d- len(erasure_slot) > 0 or l in erasure_slot or np.mod(l-1,L) in erasure_slot ) and all_known(oldPath, oldDictLostInfos) and     not (M==3 and l==1):
        if l != L-1:
                new.append( LLC.GLinkedLoop( list(oldPath) + list([-1]), messageLens, oldListLostSects + list([l]), oldDictLostInfos) ) 
        else:
            temp_path = LLC.GLinkedLoop( list(oldPath) + list([-1]), messageLens, oldListLostSects + list([l]), oldDictLostInfos)
            toKeep, updDictLostInfos = Path_goes_entry_k(d, None, L-1, temp_path, None, grand_list, messageLens, parityLens, L, M, Gis, Gijs, columns_index, sub_G_invs)
            if toKeep:
                new.append( LLC.GLinkedLoop( list(oldPath) + list([-1]) , messageLens, oldListLostSects + list([l]), updDictLostInfos) ) 
                
    return new


# Done
def final_parity_check_oop(Path, grand_list, messageLens, parityLens, L, Gijs, M):
    focusPath = Path.get_path()
    assert len(focusPath) == L
    isOkay = True

    for l in range(L):
        if focusPath[l] != -1:
            parityComputed = compute_parity_oop(L, Path, grand_list, l, messageLens, parityLens, Gijs, M)
            flag_ll = np.abs(parityComputed - grand_list[focusPath[l], l*J+messageLens[l]: (l+1)*J])
            if flag_ll.any() != 0: 
                isOkay = False
                break
    return isOkay

    

# Done
def output_message_oop(grand_list, linkedloops, L, J):
    messageLens = linkedloops[0].get_messageLens()
    n = len(linkedloops)
    msg = np.empty( (n, sum(messageLens)), dtype=int)
    for i in range(n):
        linkedloop = linkedloops[i]
        path = linkedloop.get_path()
        for l in range(L):
            if path[l] != -1: 
                msg[i, sum(messageLens[0:l]): sum(messageLens[0:l])+messageLens[l]] = grand_list[path[l], l*J: l*J+messageLens[l]]
            else: 
                msg[i, sum(messageLens[0:l]): sum(messageLens[0:l])+messageLens[l]] = linkedloop.get_dictLostInfos()[l]
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