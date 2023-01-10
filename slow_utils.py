import numpy as np
from utils import *
from binmatrix import *


def slow_compute_permissible_parity(Path,cs_decoded_tx_message,J, parityInvolved, toCheck, whichGMatrix):
    # If path length  = 2, then we wanna have parity for section 2. toCheck = 2
    parityDist = parityInvolved[:,toCheck].reshape(1,-1)[0]
    # print("----------parityDist: " + str(parityDist))
    deciders = np.nonzero(parityDist)[0] # w(decider), where decider \in deciders, decide p(toCheck) collectively
    focusPath = Path[0]

    Parity_computed = np.zeros(8, dtype=int)
    for decider in deciders:      # l labels the sections we gonna check to fix toCheck's parities
        if focusPath[decider] == -1:
            return -1 * np.ones((1,8),dtype=int)
        gen_mat = matrix_repo(dim=8)[whichGMatrix[decider][toCheck]] 
        Parity_computed = Parity_computed + np.matmul( cs_decoded_tx_message[focusPath[decider], decider*J : decider*J+8], gen_mat)
    Parity_computed = np.mod(Parity_computed, 2)

    return Parity_computed


def slow_parity_check(Parity_computed,Path,k,cs_decoded_tx_message,J,messageLengthVector, parityDistribution, useWhichMatrix):
    
    Lpath = Path.shape[1] # 當Lpath<16 也是 現在target的section 的意思
    focusPath = Path[0]
    losts = np.where( focusPath < 0 )[0]

    if Lpath < 16:  # Path 還在生長階段
        if len(losts) == 0: # 沒有 lost 最簡單的情況
            Parity = cs_decoded_tx_message[k,Lpath*J+messageLengthVector[Lpath]:(Lpath+1)*J]    # 第k行的第Lpath section的parity
            if (np.sum(np.absolute(Parity_computed-Parity)) == 0):
                return True  
            else: 
                return False 
        
        else:   # 有lost
            # 先考慮 lost 發生在很久之前
            lostSection = losts[0]          # 有且僅有一個lost
            if Lpath - lostSection > 4:     # 目前section - lostSection > 4 說明lost已經在過去被處理好了
                Parity = cs_decoded_tx_message[k,Lpath*J+messageLengthVector[Lpath]:(Lpath+1)*J]  # 第k行的第Lpath section的parity
                if (np.sum(np.absolute(Parity_computed-Parity)) == 0):
                    return True
                else: 
                    return False

            # 考慮 lost 發生在最近: 
            # if lostSection is section 4, then we gonna check
            # (w1, w2, w3, w4) => p5    # (w2, w3, w4, w5) => p6    # (w3, w4, w5, w6) => p7    # (w4, w5, w6, w7) => p8    # w5, w6, w7 and w8 are "saverSections" of lostSections
            saverSections = np.nonzero(parityDistribution[lostSection])[0]        # then saverSections = [5, 6, 7, 8]
            availSavers = [ saver for saver in saverSections if (saver <= Lpath and np.mod(saver-1,16) <= Lpath and np.mod(saver-2,16) <= Lpath and np.mod(saver-3,16) <= Lpath and np.mod(saver-4,16) <= Lpath) ]
            # print("Lpath=" + str(Lpath) + "  availSavers=" + str(availSavers) + "  lostSection=" + str(lostSection) + "  Path:" + str(focusPath))
            assert len(availSavers) > 0

            if len(availSavers) <= 1: # e.g. [5, 6]
                return True

            # 考慮至少有兩個以供驗算：
            solutions = np.empty((0,0), dtype=int)
            for saver in availSavers: # saver != lostSection
                assert saver==k or saver==np.mod(lostSection+1,16) or saver==np.mod(lostSection+2,16) or saver==np.mod(lostSection+3,16) or saver==np.mod(lostSection+4,16)
                
                row = 0
                if saver < Lpath:
                    row = focusPath[saver]
                else:
                    assert saver == Lpath
                    row = k

                parityDist = parityDistribution[:,saver].reshape(1,-1)[0]
                saverDeciders = np.nonzero(parityDist)[0]
                minuend = cs_decoded_tx_message[row, saver*J+messageLengthVector[saver] : (1+saver)*J ]  # 被減數 Aka p(saver)
                subtrahend = np.zeros(8, dtype=int) # 減數
                for saverDecider in saverDeciders:      # l labels the sections we gonna check to fix toCheck's parities
                    if (saverDecider != lostSection):
                        gen_mat = matrix_repo(dim=8)[useWhichMatrix[saverDecider][saver]] 
                        if saverDecider != Lpath:
                            subtrahend=subtrahend+np.matmul(cs_decoded_tx_message[focusPath[saverDecider],saverDecider*J:saverDecider*J+messageLengthVector[saverDecider]],gen_mat) # 都是 info * G
                        else:
                            assert saverDecider == Lpath
                            subtrahend=subtrahend+np.matmul(cs_decoded_tx_message[k,saverDecider*J:saverDecider*J+messageLengthVector[saverDecider]],gen_mat) # 都是 info * G

                subtrahend = np.mod(subtrahend, 2)
                gen_mat = matrix_repo(dim=8)[useWhichMatrix[lostSection][saver]]
                gen_binmat = BinMatrix(gen_mat)
                gen_binmat_inv = np.array(gen_binmat.inv())
                theLostPart = np.mod( np.matmul(  np.mod(minuend - subtrahend,2) , gen_binmat_inv ), 2)
                solutions = np.vstack((solutions, theLostPart)) if solutions.size else theLostPart
            
            # print(solutions)
            if np.all(solutions == solutions[0]):
                return True
            else: 
                return False

    else: # Path is already full
        if len(losts) >0: # 有lost的
            lostSection = losts[0]
            Recovered_info = -1 * np.ones((1,8),dtype=int)
            solutions = np.empty((0,0), dtype=int)
            saverSections = np.nonzero(parityDistribution[lostSection])[0]
            for saver in saverSections:         # if lostSection = 12, saverSections = [13, 14, 15, 16]
                parityDist = parityDistribution[:,saver].reshape(1,-1)[0]
                saverDeciders = np.nonzero(parityDist)[0]   # if saver = 13, saverDeciders = [9, 10, 11, 12]
                minuend = cs_decoded_tx_message[ focusPath[saver], saver*J+messageLengthVector[saver] : (1+saver)*J ]  # 被減數 Aka p(saver)
                subtrahend = np.zeros(8, dtype=int) # 減數
                for saverDecider in saverDeciders:      # l labels the sections we gonna check to fix toCheck's parities
                    if (saverDecider != lostSection):
                        gen_mat = matrix_repo(dim=8)[useWhichMatrix[saverDecider][saver]] 
                        subtrahend=subtrahend+np.matmul(cs_decoded_tx_message[ focusPath[saverDecider], saverDecider*J:saverDecider*J+messageLengthVector[saverDecider]],gen_mat)
                
                subtrahend = np.mod(subtrahend, 2)
                gen_mat = matrix_repo(dim=8)[useWhichMatrix[lostSection][saver]] 
                gen_binmat = BinMatrix(gen_mat)
                gen_binmat_inv = np.array(gen_binmat.inv())
                theLostPart = np.mod( np.matmul(  np.mod(minuend - subtrahend,2) , gen_binmat_inv ), 2)
                solutions = np.vstack((solutions, theLostPart)) if solutions.size else theLostPart
            
            if np.all(solutions == solutions[0]):
                Recovered_info = solutions[0]
            else: 
                # print("因為沒有解，不行")
                return False

            # 走到這裡的一定已經算過 info了
            for ll in range(16): # 每個section都直接算一遍
                if ll!=lostSection:
                    llParityDist = parityDistribution[:,ll].reshape(1,-1)[0]
                    llDeciders = np.nonzero(llParityDist)[0]
                    Parity_computed_ll = np.zeros((1,8),dtype=int)
                    for llDecider in llDeciders:
                        gen_mat = matrix_repo(dim=8)[useWhichMatrix[llDecider][ll]] 
                        if llDecider == lostSection:
                            Parity_computed_ll = Parity_computed_ll + np.matmul(Recovered_info, gen_mat)
                        else: 
                            Parity_computed_ll = Parity_computed_ll + np.matmul(cs_decoded_tx_message[Path[0][llDecider], llDecider*J:llDecider*J+8], gen_mat)
                    Parity_computed_ll = np.mod(Parity_computed_ll, 2)
                    flag_ll = sum( np.abs(Parity_computed_ll.reshape(-1) - cs_decoded_tx_message[Path[0][ll], ll*J+messageLengthVector[ll]: (ll+1)*J].reshape(-1) ))
                    if flag_ll!=0:
                        # print("一條完整(有lost)的path 在這個section出錯" + str(ll))
                        return False
            return True
        else:
            # 沒有lost的
            for ll in range(16): # 每個section都直接算一遍
                llParityDist = parityDistribution[:,ll].reshape(1,-1)[0]
                llDeciders = np.nonzero(llParityDist)[0]
                Parity_computed_ll = np.zeros((1,8),dtype=int)
                for llDecider in llDeciders:
                    gen_mat = matrix_repo(dim=8)[useWhichMatrix[llDecider][ll]] 
                    Parity_computed_ll = Parity_computed_ll + np.matmul(cs_decoded_tx_message[Path[0][llDecider], llDecider*J:llDecider*J+8], gen_mat)
                Parity_computed_ll = np.mod(Parity_computed_ll, 2)
                flag_ll = sum( np.abs(Parity_computed_ll.reshape(-1) - cs_decoded_tx_message[Path[0][ll], ll*J+messageLengthVector[ll]: (ll+1)*J].reshape(-1) ))
                if flag_ll!=0:
                    # print("一條完(沒lost)的path 在這個section出錯" + str(ll))
                    return False
            return True





def slow_recover_msg(sectionLost, decoded_message, parityDistribution, messageLengthVector, J, L, useWhichMatrix):
    recovered_msg = np.array([], dtype= int).reshape(1,-1)
    lostSection = sectionLost[0]
    for ll in np.arange(L):
        if ll not in sectionLost:
            recovered_msg = np.concatenate( (recovered_msg, decoded_message[0][ ll*J : ll*J+messageLengthVector[ll] ].reshape(1,-1)[0]) , axis=None )
        else: # ll  in sectoinLost:              # suppose ll = 5. we first check section 5 determines what? 
            saverSections = np.nonzero(parityDistribution[ll])[0]        # then saverSections = [6, 7, 8, 9]
            solutions = np.empty((0,0), dtype=int)

            for saver in saverSections:
                parityDist = parityDistribution[:,saver].reshape(1,-1)[0]
                saverDeciders = np.nonzero(parityDist)[0]
                minuend =  decoded_message[0][ saver*J+messageLengthVector[saver] : (1+saver)*J ].reshape(1,-1)[0]

                subtrahend = np.zeros(8, dtype=int)
                for saverDecider in saverDeciders:      # l labels the sections we gonna check to fix toCheck's parities
                    if (saverDecider != lostSection):
                        gen_mat = matrix_repo(dim=8)[useWhichMatrix[saverDecider][saver]] 
                        subtrahend = subtrahend + np.matmul( decoded_message[0, saverDecider*J : saverDecider*J+8], gen_mat)
                
                subtrahend = np.mod(subtrahend, 2)
                gen_mat = matrix_repo(dim=8)[useWhichMatrix[lostSection][saver]] 
                gen_binmat = BinMatrix(gen_mat)
                gen_binmat_inv = np.array(gen_binmat.inv())
                theLostPart = np.mod( np.matmul(  np.mod(minuend - subtrahend,2) , gen_binmat_inv ), 2)
                solutions = np.vstack((solutions, theLostPart)) if solutions.size else theLostPart

            # print(" -------- ")
            # print(solutions)
            
            if np.all(solutions == solutions[0]):
                recovered_msg = np.concatenate( (recovered_msg, theLostPart) , axis=None)
                print(" ** This candidate is valid.")
    
    return recovered_msg.reshape(1,-1)
