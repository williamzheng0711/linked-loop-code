import numpy as np
from utils import *
from binmatrix import *

def slow_compute_permissible_parity(Path,cs_decoded_tx_message,J, parityInvolved, toCheck, whichGMatrix, parityLen, messageLen):
    # If path length  = 2, then we wanna have parity for section 2. toCheck = 2
    parityDist = parityInvolved[:,toCheck].reshape(1,-1)[0]
    # print("----------parityDist: " + str(parityDist), "toCheck: " + str(toCheck))
    deciders = np.nonzero(parityDist)[0] # w(decider), where decider \in deciders, decide p(toCheck) collectively
    focusPath = Path[0]

    Parity_computed = np.zeros(parityLen, dtype=int)
    for decider in deciders:      # l labels the sections we gonna check to fix toCheck's parities
        if focusPath[decider] == -1:
            return -1 * np.ones((1,parityLen),dtype=int)
        assert whichGMatrix[decider][toCheck] != -1
        gen_mat = matrix_repo(dim=messageLen)[ whichGMatrix[decider][toCheck] ] 
        Parity_computed = Parity_computed + np.matmul( cs_decoded_tx_message[focusPath[decider], decider*J : decider*J+messageLen], gen_mat)
    Parity_computed = np.mod(Parity_computed, 2)

    return Parity_computed

def slow_parity_check(Parity_computed,Path,k,cs_decoded_tx_message,J,messageLen,parityLen, parityDistribution, useWhichMatrix, L, windowSize):
    Lpath = Path.shape[1] # 當Lpath<16 也是 現在target的section 的意思
    focusPath = Path[0]
    losts = np.where( focusPath < 0 )[0]

    generator_matrices = matrix_repo(dim=messageLen)
    inv_generator_matrices = matrix_inv_repo(dim=messageLen)

    if Lpath < L:  # Path 還在生長階段
        if len(losts) == 0: # 沒有 lost 最簡單的情況
            Parity = cs_decoded_tx_message[k,Lpath*J+messageLen:(Lpath+1)*J]    # 第k行的第Lpath section的parity
            if (np.sum(np.absolute(Parity_computed-Parity)) == 0):
                return True  
            else: 
                return False 
        
        else:   # 有lost
            # 先考慮 lost 發生在很久之前
            lostSection = losts[0]          # 有且僅有一個lost
            if Lpath - lostSection > windowSize:     # 目前section - lostSection > 4 說明lost已經在過去被處理好了
                Parity = cs_decoded_tx_message[k,Lpath*J+messageLen:(Lpath+1)*J]  # 第k行的第Lpath section的parity
                if (np.sum(np.absolute(Parity_computed-Parity)) == 0):
                    return True
                else: 
                    return False

            # 考慮 lost 發生在最近: # if lostSection is section 4, then we gonna check
            # (w1, w2, w3, w4) => p5    # (w2, w3, w4, w5) => p6    # (w3, w4, w5, w6) => p7    # (w4, w5, w6, w7) => p8    # w5, w6, w7 and w8 are "saverSections" of lostSections
            saverSections = np.nonzero(parityDistribution[lostSection])[0]        # then saverSections = [5, 6, 7, 8]
            availSavers = [saver for saver in saverSections if np.array([np.mod(saver-x,L)<=Lpath for x in range(windowSize+1)]).all() == True ]
        
            assert len(availSavers) > 0
            if len(availSavers) <= 1:  # Because we need at least TWO results to compare.
                return True

            solutions = np.empty((0,0), dtype=int)
            for saver in availSavers: # saver != lostSection
                assert np.array([saver == np.mod(saver+x,L) for x in range(windowSize+1)]).any() == True
                row = 0
                if saver < Lpath: 
                    row = focusPath[saver]
                else: 
                    assert saver == Lpath; 
                    row = k

                parityDist = parityDistribution[:,saver].reshape(1,-1)[0]
                saverDeciders = np.nonzero(parityDist)[0]
                minuend = cs_decoded_tx_message[row, saver*J+messageLen: (1+saver)*J ]  # 被減數 Aka p(saver)
                subtrahend = np.zeros(parityLen, dtype=int) # 減數
                for saverDecider in saverDeciders:      # l labels the sections we gonna check to fix toCheck's parities
                    if (saverDecider != lostSection):
                        gen_mat = generator_matrices[useWhichMatrix[saverDecider][saver]] 
                        if saverDecider != Lpath:
                            subtrahend=subtrahend+np.matmul(cs_decoded_tx_message[focusPath[saverDecider],saverDecider*J:saverDecider*J+messageLen],gen_mat) # 都是 info * G
                        else:
                            assert saverDecider == Lpath
                            subtrahend=subtrahend+np.matmul(cs_decoded_tx_message[k,saverDecider*J:saverDecider*J+messageLen],gen_mat) # 都是 info * G

                subtrahend = np.mod(subtrahend, 2)
                gen_mat = generator_matrices[useWhichMatrix[lostSection][saver]]
                gen_binmat_inv = np.array(inv_generator_matrices[useWhichMatrix[lostSection,saver]])
                theLostPart = np.mod(np.matmul(np.mod(minuend - subtrahend,2),gen_binmat_inv),2)
                solutions = np.vstack((solutions,theLostPart)) if solutions.size else theLostPart
            
            if np.all(solutions == solutions[0]): 
                # print("對的")
                return True
            else:
                # print("不對") 
                return False

    else: # Path is already full
        lostSection = losts[0] if len(losts)>0 else -1
        Recovered_info = -1 * np.ones((1,messageLen),dtype=int)
        solutions = np.empty((0,0), dtype=int)
        saverSections = np.nonzero(parityDistribution[lostSection])[0]
        for saver in saverSections:         # if lostSection = 12, saverSections = [13, 14, 15, 16]
            parityDist = parityDistribution[:,saver].reshape(1,-1)[0]
            saverDeciders = np.nonzero(parityDist)[0]   # if saver = 13, saverDeciders = [9, 10, 11, 12]
            minuend = cs_decoded_tx_message[focusPath[saver],saver*J+messageLen:(1+saver)*J]  # 被減數 Aka p(saver)
            subtrahend = np.zeros(parityLen, dtype=int) # 減數
            for saverDecider in saverDeciders:      # l labels the sections we gonna check to fix toCheck's parities
                if (saverDecider != lostSection):
                    gen_mat = generator_matrices[useWhichMatrix[saverDecider][saver]] 
                    subtrahend = subtrahend+np.matmul(cs_decoded_tx_message[focusPath[saverDecider],saverDecider*J:saverDecider*J+messageLen] , gen_mat)
            
            subtrahend= np.mod(subtrahend,2)
            gen_mat= generator_matrices[useWhichMatrix[lostSection][saver]] 
            gen_binmat_inv= np.array(inv_generator_matrices[useWhichMatrix[lostSection,saver]])
            theLostPart= np.mod(np.matmul(np.mod(minuend-subtrahend,2),gen_binmat_inv),2)
            solutions= np.vstack((solutions,theLostPart)) if solutions.size else theLostPart
        
        if np.all(solutions == solutions[0]): 
            Recovered_info = solutions[0]
        else: 
            return False

        for ll in range(L): # 每個section都直接算一遍
            if ll!=lostSection:
                llParityDist = parityDistribution[:,ll].reshape(1,-1)[0]
                llDeciders = np.nonzero(llParityDist)[0]
                Parity_computed_ll= np.zeros((1,parityLen),dtype=int)
                for llDecider in llDeciders:
                    gen_mat= generator_matrices[useWhichMatrix[llDecider][ll]] 
                    if llDecider == lostSection:
                        Parity_computed_ll= Parity_computed_ll+np.matmul(Recovered_info, gen_mat)
                    else: 
                        Parity_computed_ll= Parity_computed_ll+np.matmul(cs_decoded_tx_message[Path[0][llDecider], llDecider*J:llDecider*J+messageLen], gen_mat)
                Parity_computed_ll = np.mod(Parity_computed_ll,2)
                flag_ll = sum( np.abs(Parity_computed_ll.reshape(-1) - cs_decoded_tx_message[Path[0][ll], ll*J+messageLen:(ll+1)*J].reshape(-1)))
                if flag_ll!=0:
                    return False

        # if lostSection == -1: print("No lost at all.")
        return True


def slow_recover_msg(sectionLost, decoded_message, parityDistribution, messageLen, parityLen, J, L, useWhichMatrix):
    generator_matrices = matrix_repo(dim=messageLen)
    inv_generator_matrices = matrix_inv_repo(dim=messageLen)
    recovered_msg = np.array([], dtype= int).reshape(1,-1)
    lostSection = sectionLost[0]
    for ll in np.arange(L):
        if ll not in sectionLost:
            recovered_msg = np.concatenate( (recovered_msg, decoded_message[0][ ll*J : ll*J+messageLen ].reshape(1,-1)[0]) , axis=None )
        else: # ll  in sectoinLost:              # suppose ll = 5. we first check section 5 determines what? 
            saverSections = np.nonzero(parityDistribution[ll])[0]        # then saverSections = [6, 7, 8, 9]
            solutions = np.empty((0,0), dtype=int)

            for saver in saverSections:
                parityDist = parityDistribution[:,saver].reshape(1,-1)[0]
                saverDeciders = np.nonzero(parityDist)[0]
                minuend =  decoded_message[0][ saver*J+messageLen: (1+saver)*J ].reshape(1,-1)[0]
                subtrahend = np.zeros(parityLen, dtype=int)
                for saverDecider in saverDeciders:      # l labels the sections we gonna check to fix toCheck's parities
                    if (saverDecider != lostSection):
                        gen_mat = generator_matrices[useWhichMatrix[saverDecider][saver]] 
                        subtrahend = subtrahend + np.matmul( decoded_message[0, saverDecider*J : saverDecider*J+messageLen], gen_mat)
                
                subtrahend = np.mod(subtrahend, 2)
                gen_mat = generator_matrices[useWhichMatrix[lostSection][saver]] 
                gen_binmat_inv = np.array(inv_generator_matrices[useWhichMatrix[lostSection, saver]])
                theLostPart = np.mod( np.matmul( np.mod(minuend - subtrahend,2) , gen_binmat_inv), 2)
                solutions = np.vstack((solutions, theLostPart)) if solutions.size else theLostPart
            
            if np.all(solutions == solutions[0]):
                recovered_msg = np.concatenate( (recovered_msg, theLostPart) , axis=None)
                # print(" | This candidate is valid.")
    
    return recovered_msg.reshape(1,-1)

def slow_decode_deal_with_root_i(i,L,cs_decoded_tx_message, J,parityInvolved, whichGMatrix, messageLen, listSize, parityLen, windowSize):
    # Every i is a root.
    # If section ZERO contains -1, then this root is defective
    if cs_decoded_tx_message[i,0] == -1:
        # print("i= " + str(i)+" 是-1")
        return -1*np.ones((1,messageLen * L), dtype=int)
    
    # This root is not defective.
    # print("不是-1 i=" + str(i))
    Paths = np.array([[i]])
    for l in range(1, L):
        # Grab the parity generator matrix corresponding to this section  
        new=np.empty( shape=(0,0))
        for j in range(Paths.shape[0]):
            Path=Paths[j].reshape(1,-1)     
            Parity_computed= np.ones((1,parityLen),dtype=int)
            if l >= windowSize:
                Parity_computed = slow_compute_permissible_parity(Path,cs_decoded_tx_message,J,parityInvolved,l,whichGMatrix,parityLen,messageLen)
            for k in range(listSize):
                if cs_decoded_tx_message[k, l*J] != -1:
                    index = l < windowSize or slow_parity_check(Parity_computed,Path,k,cs_decoded_tx_message,J,messageLen,parityLen,parityInvolved,whichGMatrix,L,windowSize)
                    if index: # If parity constraints are satisfied, update the path
                        new = np.vstack((new,np.hstack((Path.reshape(1,-1),np.array([[k]]))))) if new.size else np.hstack((Path.reshape(1,-1),np.array([[k]])))
        Paths = new 
        if Paths.shape[0] == 0:
            break
    
    # Let us go to check section 0, 1, 2 and 3. They are not checked in above.
    PathsUpdated = np.empty( shape=(0,0))
    for j in range(Paths.shape[0]):
        isOkay = True
        Path = Paths[j].reshape(1,-1)
        for ll in range(windowSize):
            Parity_computed_ll = slow_compute_permissible_parity(Path,cs_decoded_tx_message,J, parityInvolved, ll, whichGMatrix, parityLen, messageLen)
            flag_ll = sum( np.abs(Parity_computed_ll - cs_decoded_tx_message[Path[0][ll], ll*J+messageLen: (ll+1)*J]) )
            if flag_ll !=0: 
                isOkay = False; break
        if isOkay:
            PathsUpdated = np.vstack((PathsUpdated, Path)) if PathsUpdated.size else Path
    Paths = PathsUpdated

    # Handle multiple valid paths
    if Paths.shape[0] >= 1:  
        if Paths.shape[0] >= 2:
            flag = check_if_identical_msgs(Paths, cs_decoded_tx_message, L,J,messageLen)
            if flag:
                return extract_msg_bits(Paths[0].reshape(1,-1),cs_decoded_tx_message, L,J,messageLen)
            else:
                # optimalOne = 0
                # pathVar = np.zeros((Paths.shape[0]))
                # for whichPath in range(Paths.shape[0]):
                #     fadingValues = []
                #     for l in range(Paths.shape[1]):     
                #         fadingValues.append( sigValues[ Paths[whichPath][l] ][l] ) 
                #     pathVar[whichPath] = np.var(fadingValues)
                # optimalOne = np.argmin(pathVar)
                return extract_msg_bits(Paths.reshape(Paths.shape[0],-1),cs_decoded_tx_message, L,J,messageLen)
        elif Paths.shape[0] == 1:
            return extract_msg_bits(Paths.reshape(1,-1),cs_decoded_tx_message, L,J,messageLen)
    
    return -1*np.ones((1,messageLen * L), dtype=int)

def slow_correct_each_section_and_path(l, j, Paths, cs_decoded_tx_message, J, parityInvolved, whichGMatrix, listSize, messageLen, parityLen, L, windowSize):
    new = np.empty( shape=(0,0), dtype=int)
    Path = Paths[j].reshape(1,-1)
    pathArgNa = np.where( Path[0] < 0 )[0]    

    Parity_computed = -1 * np.ones((1,parityLen),dtype=int)
    if l >= windowSize: 
        Parity_computed = slow_compute_permissible_parity(Path, cs_decoded_tx_message, J, parityInvolved, l, whichGMatrix, parityLen, messageLen)
    for k in range(listSize):
        if cs_decoded_tx_message[k,l*J] != -1:
            if l < windowSize:
                new = np.vstack((new,np.hstack((Path.reshape(1,-1),np.array([[k]]))))) if new.size else np.hstack((Path.reshape(1,-1),np.array([[k]])))
            else :  # now l >= 4:
                index = slow_parity_check(Parity_computed, Path, k, cs_decoded_tx_message, J, messageLen,parityLen,parityInvolved, whichGMatrix, L, windowSize) 
                if index:
                    new = np.vstack((new,np.hstack((Path.reshape(1,-1),np.array([[k]]))))) if new.size else np.hstack((Path.reshape(1,-1),np.array([[k]])))
    if len(pathArgNa) == 0:
        new = np.vstack((new,np.hstack((Path.reshape(1,-1),np.array([[-1]]))))) if new.size else np.hstack((Path.reshape(1,-1),np.array([[-1]])))
    return new



