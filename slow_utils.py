import numpy as np
from utils import *
from binmatrix import *


def slow_compute_permissible_parity(Path,cs_decoded_tx_message,J, parityDistribution, toCheck, useWhichMatrix, pathDict):
    # if path length  = 2
    # then we wanna have parity for section 2. section2Check = 2
    parityDist = parityDistribution[:,toCheck].reshape(1,-1)[0]
    # print("----------parityDist: " + str(parityDist))
    deciders = np.nonzero(parityDist)[0]
    # print("parityDependents = " + str(parityDependents))
    focusPath = Path[0]

    # Parity_computed = np.zeros(8, dtype=int)
    # keyToFind = "#".join([str(item) for item in focusPath[0:len(focusPath)-1]])
    # if toCheck == 4 and keyToFind in pathDict: # This means we only need to add the last new node of path
    #     decider = deciders[-1]
    #     gen_mat = matrix_repo(dim=8)[useWhichMatrix[decider][toCheck]] # only last deciders is needed
    #     Parity_computed = pathDict[keyToFind] + np.matmul( cs_decoded_tx_message[focusPath[decider], decider*J : decider*J+8], gen_mat)
    #     # print("來了來了")

    # Parity_computed = np.mod(Parity_computed, 2)

    # # else:

    Parity_computed2 = np.zeros(8, dtype=int)

    for decider in deciders:      # l labels the sections we gonna check to fix toCheck's parities
        gen_mat = matrix_repo(dim=8)[useWhichMatrix[decider][toCheck]] 
        Parity_computed2 = Parity_computed2 + np.matmul( cs_decoded_tx_message[focusPath[decider], decider*J : decider*J+8], gen_mat)

    Parity_computed2 = np.mod(Parity_computed2, 2)
    # if toCheck==4: print("Parity_computed: " + str(Parity_computed) + " and Parity_computed2 " + str(Parity_computed2))
    newAddedKey = "#".join([str(item) for item in focusPath[0:len(focusPath)]])
    pathDict[newAddedKey] = Parity_computed2

    return Parity_computed2, pathDict


def slow_parity_check(Parity_computed,Path,k,cs_decoded_tx_message,J,messageLengthVector, parityDistribution, useWhichMatrix):
    
    Lpath = Path.shape[1]
    focusPath = Path[0]
    losts = np.where( focusPath < 0 )[0]

    if Lpath < 16:  # Path 還在生長階段
        if len(losts) == 0: # 沒有 lost 最簡單的情況
            Parity = cs_decoded_tx_message[k,Lpath*J+messageLengthVector[Lpath]:(Lpath+1)*J]
            if (np.sum(np.absolute(Parity_computed-Parity)) == 0):
                return True  
        
        else:   # 有lost
            # 先考慮 lost 發生在很久之前
            lostSection = losts[0]
            if Lpath - lostSection > 4:     # 目前section - lostSection > 4 說明lost已經在過去被處理好了
                Parity = cs_decoded_tx_message[k,Lpath*J+messageLengthVector[Lpath]:(Lpath+1)*J]
                if (np.sum(np.absolute(Parity_computed-Parity)) == 0):
                    return True

            # lost 發生在最近: 

            # if lostSection is section 4, then we gonna check
            # (w1, w2, w3, w4) => p5    # (w2, w3, w4, w5) => p6    # (w3, w4, w5, w6) => p7    # (w4, w5, w6, w7) => p8
            # w5, w6, w7 and w8 are "saverSections" of lostSections
            saverSections = np.nonzero(parityDistribution[lostSection])[0]        # then saverSections = [5, 6, 7, 8]
            availSavers = [ saver for saver in saverSections if (saver <= Lpath and np.mod(saver-1,16) <= Lpath) ]
            
            if len(availSavers) <= 1: # e.g. [5, 6]
                return True

            # 考慮至少有兩個以供驗算：
            solutions = np.empty((0,0), dtype=int)
            for saver in availSavers:

                row = focusPath[saver] if saver < Lpath else k

                parityDist = parityDistribution[:,saver].reshape(1,-1)[0]
                saverDeciders = np.nonzero(parityDist)[0]
                minuend = cs_decoded_tx_message[row, saver*J+messageLengthVector[saver] : (1+saver)*J ]  # 被減數 Aka p(saver)
                subtrahend = np.zeros(8, dtype=int) # 減數
                for saverDecider in saverDeciders:      # l labels the sections we gonna check to fix toCheck's parities
                    if (saverDecider != lostSection):
                        gen_mat = matrix_repo(dim=8)[useWhichMatrix[saverDecider][saver]] 
                        subtrahend=subtrahend+np.matmul(cs_decoded_tx_message[row, saverDecider*J:saverDecider*J+messageLengthVector[saverDecider]],gen_mat)
                
                subtrahend = np.mod(subtrahend, 2)
                gen_binmat = BinMatrix(gen_mat)
                gen_binmat_inv = np.array(gen_binmat.inv())
                theLostPart = np.mod( np.matmul(  np.mod(minuend - subtrahend,2) , gen_binmat_inv ), 2)
                solutions = np.vstack((solutions, theLostPart)) if solutions.size else theLostPart

            if np.all(solutions == solutions[0]):
                return True
            else: 
                return False

    else: # Path is already full
        needHandleLost = True
        lostSection = losts[0] # can only be 12 or 13 or 14 or 15
        if lostSection < 11: needHandleLost = False

        if needHandleLost: # 問題出在 12 13 14 15
            availSavers = np.nonzero(parityDistribution[lostSection])[0]       
            solutions = np.empty((0,0), dtype=int)
            for saver in availSavers:
                parityDist = parityDistribution[:,saver].reshape(1,-1)[0]
                saverDeciders = np.nonzero(parityDist)[0]
                minuend = cs_decoded_tx_message[focusPath[saver], saver*J+messageLengthVector[saver] : (1+saver)*J ].reshape(1,-1)[0]   

                subtrahend = np.zeros(8, dtype=int)
                for saverDecider in saverDeciders:      # l labels the sections we gonna check to fix toCheck's parities
                    if (saverDecider != lostSection):
                        gen_mat = matrix_repo(dim=8)[useWhichMatrix[saverDecider][saver]] 
                        subtrahend = subtrahend + np.matmul( cs_decoded_tx_message[focusPath[saver], saverDecider*J : saverDecider*J+8], gen_mat)
                
                subtrahend = np.mod(subtrahend, 2)
                gen_binmat = BinMatrix(gen_mat)
                gen_binmat_inv = np.array(gen_binmat.inv())
                theLostPart = np.mod( np.matmul(  np.mod(minuend - subtrahend,2) , gen_binmat_inv ), 2)
                solutions = np.vstack((solutions, theLostPart)) if solutions.size else theLostPart

            if np.all(solutions == solutions[0])==False:
                return False

        isOkay = True
        for ll in [0,1,2,3]:
            llParityDist = parityDistribution[:,ll].reshape(1,-1)[0]
            llDeciders = np.nonzero(llParityDist)[0]
            if (lostSection in llDeciders) == False:
                Parity_computed_ll, _ = slow_compute_permissible_parity(Path,cs_decoded_tx_message,J, parityDistribution, ll, useWhichMatrix, _)
                flag_ll = sum( np.abs(Parity_computed_ll - cs_decoded_tx_message[Path[0][ll], ll*J+messageLengthVector[ll]: (ll+1)*J]) )
                if flag_ll !=0: 
                    isOkay = False
                    return False

        if isOkay == True:
            return True

    return False



def slow_recover_msg(sectionLost, decoded_message, parityDistribution, messageLengthVector, J, L, useWhichMatrix):
    recovered_msg = np.array([], dtype= int).reshape(1,-1)
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
                    if (saverDecider != sectionLost):
                        gen_mat = matrix_repo(dim=8)[useWhichMatrix[saverDecider][saver]] 
                        subtrahend = subtrahend + np.matmul( decoded_message[0, saverDecider*J : saverDecider*J+8], gen_mat)
                
                subtrahend = np.mod(subtrahend, 2)
                gen_binmat = BinMatrix(gen_mat)
                gen_binmat_inv = np.array(gen_binmat.inv())
                theLostPart = np.mod( np.matmul(  np.mod(minuend - subtrahend,2) , gen_binmat_inv ), 2)
                
                solutions = np.vstack((solutions, theLostPart)) if solutions.size else theLostPart

            print(" -------- ")
            print(solutions)
            print(decoded_message)
            
            if np.all(solutions == solutions[0]):
                recovered_msg = np.concatenate( (recovered_msg, theLostPart) , axis=None)
    
    return recovered_msg.reshape(1,-1)
