import numpy as np
from utils import *
from slow_utils import *

def Slow_encode(tx_message,K,L,J,P,Ml,messageLengthVector,parityLengthVector, parityDistribution, useWhichMatrix):
    encoded_tx_message = np.zeros((K,Ml+P),dtype=int)
    # write the info bits into corresponding positions of each section
    encoded_tx_message[:,0:messageLengthVector[0]] = tx_message[:,0:messageLengthVector[0]]
    info = []
    for i in range(0,L):
        info.append(tx_message[:,np.sum(messageLengthVector[0:i]):np.sum(messageLengthVector[0:i+1])])
        encoded_tx_message[:,i*J:i*J+messageLengthVector[i]] = info[i]
    
    info = np.array(info) # Pls note that info is 100 x 8 size

    for i in np.arange(0,L,1):
        whoDecidesI = np.nonzero(np.transpose(parityDistribution)[i])[0]
        # print("i=" + str(i) + " whoDecidesI=" + str(whoDecidesI))     # i=0 whoDecidesI=[12 13 14 15]
        parity_i = np.zeros((K, parityLengthVector[i]), dtype=int )
        for decider in whoDecidesI:
            parity_i = parity_i + np.matmul( info[decider], matrix_repo(dim=8)[useWhichMatrix[decider][i]] )
        encoded_tx_message[:, i*J+np.sum(messageLengthVector[i]) : (i+1)*J] = np.mod(parity_i, 2)

    # One can check what a outer-encoded message looks like in the csv file.
    np.savetxt('encoded_message.csv', encoded_tx_message[0].reshape(16,16), delimiter=',', fmt='%d')

    return encoded_tx_message


def Slow_decoder_fader(decBetaNoised, decBetaPos, L,J,B,parityLengthVector,messageLengthVector,listSize, parityDistribution, useWhichMatrix):
    # decBetaSignificants size is (listSize, 16)
    cs_decoded_tx_message = np.zeros( (listSize, L*J) ) # (listSize, 256)
    for id_row in range(decBetaPos.shape[0]):
        for id_col in range(decBetaPos.shape[1]):
            a = np.binary_repr(decBetaPos[id_row][id_col], width=J)      # print("a = " + str(a))
            b = np.array([int(n) for n in a] ).reshape(1,-1)             # print("b = " + str(b))
            cs_decoded_tx_message[ id_row, id_col*J: (id_col+1)*J ] = b[0, 0:J]

    # decBetaNoised shape is (listSize, 16)
    listSizeOrder = np.flip(np.argsort( decBetaNoised[:,0] ))
    tree_decoded_tx_message = np.empty(shape=(0,0))
    usedRootsIndex = []
    pathDict = {}   # an dictionary to look up calculated sub-paths' permissible parity

    for i, arg_i in zip(listSizeOrder, np.arange(len(listSizeOrder))):
        Paths = np.array([[i]])
        # not tail bit yet
        for l in range(1, L):
            # Grab the parity generator matrix corresponding to this section  
            new=np.empty( shape=(0,0))
            for j in range(Paths.shape[0]):
                Path=Paths[j].reshape(1,-1) 
                Parity_computed= np.ones((1,8),dtype=int)
                if l >= 4:
                    Parity_computed, pathDict = slow_compute_permissible_parity(Path, cs_decoded_tx_message, J, parityDistribution, l, useWhichMatrix, pathDict)
                # print("Parity_computed is: " + str(Parity_computed) )
                for k in range(listSize):
                    # Verify parity constraints for the children of surviving path
                    index = l<4 or slow_parity_check(Parity_computed, Path, k, cs_decoded_tx_message, J, messageLengthVector, parityDistribution, useWhichMatrix)
                    # If parity constraints are satisfied, update the path
                    if index:
                        new = np.vstack((new,np.hstack((Path.reshape(1,-1),np.array([[k]]))))) if new.size else np.hstack((Path.reshape(1,-1),np.array([[k]])))
            Paths = new 
            # print("l=" + str(l) + ' path number: ' + str(Paths.shape[0]))
            if Paths.shape[0] == 0:
                break
        
        # let's do tail biting!!!
        # print("跑到這了 有多少條path？" + str(Paths.shape[0]))
        PathsUpdated = np.empty( shape=(0,0))
        for j in range(Paths.shape[0]):
            isOkay = True
            Path = Paths[j].reshape(1,-1)
            for ll in [0,1,2,3]:
                Parity_computed_ll, pathDict = slow_compute_permissible_parity(Path,cs_decoded_tx_message,J, parityDistribution, ll, useWhichMatrix, pathDict)
                flag_ll = sum( np.abs(Parity_computed_ll - cs_decoded_tx_message[Path[0][ll], ll*J+messageLengthVector[ll]: (ll+1)*J]) )
                if flag_ll !=0: 
                    isOkay = False
                    break
            if isOkay:
                PathsUpdated = np.vstack((PathsUpdated, Path)) if PathsUpdated.size else Path
            # print("path " + str(j) + " " + str(isOkay))
        Paths = PathsUpdated
        # print("跑到這了!!!")


        if Paths.shape[0] >= 1:  
            if Paths.shape[0] >= 2:
                flag = check_if_identical_msgs(Paths, cs_decoded_tx_message, L,J,parityLengthVector,messageLengthVector)
                if flag:
                    tree_decoded_tx_message = np.vstack((tree_decoded_tx_message,extract_msg_bits(Paths[0].reshape(1,-1),cs_decoded_tx_message, L,J,parityLengthVector,messageLengthVector))) if tree_decoded_tx_message.size else extract_msg_bits(Paths[0].reshape(1,-1),cs_decoded_tx_message, L,J,parityLengthVector,messageLengthVector)
                else:
                    optimalOne = 0
                    pathVar = np.zeros((Paths.shape[0]))
                    for whichPath in range(Paths.shape[0]):
                        fadingValues = []
                        for l in range(Paths.shape[1]):     # decBetaSignificantsPos size is (listSize, 16)s
                            fadingValues.append( decBetaNoised[ Paths[whichPath][l] ][l] )        # print("fadingValues = " + str(fadingValues))
                        demeanFading = fadingValues - np.mean(fadingValues)
                        pathVar[whichPath] = np.var(fadingValues)
                    optimalOne = np.argmin(pathVar)
                    tree_decoded_tx_message = np.vstack((tree_decoded_tx_message,extract_msg_bits(Paths[optimalOne].reshape(1,-1),cs_decoded_tx_message, L,J,parityLengthVector,messageLengthVector))) if tree_decoded_tx_message.size else extract_msg_bits(Paths[optimalOne].reshape(1,-1),cs_decoded_tx_message, L,J,parityLengthVector,messageLengthVector)
            elif Paths.shape[0] == 1:
                tree_decoded_tx_message = np.vstack((tree_decoded_tx_message,extract_msg_bits(Paths.reshape(1,-1),cs_decoded_tx_message, L,J,parityLengthVector,messageLengthVector))) if tree_decoded_tx_message.size else extract_msg_bits(Paths.reshape(1,-1),cs_decoded_tx_message, L,J,parityLengthVector,messageLengthVector)

            usedRootsIndex.append(i)     
            # print(str(arg_i) + " 又跑完了一個root" + "  ******* 有一條消息 *******")   

    return tree_decoded_tx_message, usedRootsIndex






def Slow_corrector_fader(decBetaNoised, decBetaPos,L,J,B,parityLengthVector,messageLengthVector,listSize, parityDistribution, usedRootsIndex, useWhichMatrix):
    # decBetaSignificants size is (listSize, 16)
    cs_decoded_tx_message = np.zeros( (listSize, L*J) ) # (listSize, 256)
    for id_row in range(decBetaPos.shape[0]):
        for id_col in range(decBetaPos.shape[1]):
            a = np.binary_repr(decBetaPos[id_row][id_col], width=J)
            b = np.array([int(n) for n in a] ).reshape(1,-1)
            cs_decoded_tx_message[ id_row, id_col*J: (id_col+1)*J ] = b[0, 0:J]

    # decBetaNoised shape is (listSize, 16)
    listSizeOrder = np.flip(np.argsort( decBetaNoised[:,0] )) # flip 之後就是從大到小
    listSizeOrder_remained = [x for x in listSizeOrder if x not in usedRootsIndex] # 去掉之前用過的root

    tree_decoded_tx_message = np.empty(shape=(0,0))

    targetingSections = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,0]

    for i, arg_i in zip(listSizeOrder_remained, np.arange(len(listSizeOrder_remained))):
        Paths = np.array([[i]])
        for l, arg_l in zip( targetingSections, range(len(targetingSections))):
            # print("under " + str(arg_i) + " l=" + str(l))
            # print( "Target section: " + str(l) + " | No. of paths: " + str(Paths.shape[0]) + " | How many contains -1: " + str(sum([1 for Path in Paths if np.any(Path<0)])) )
            if Paths.shape[1] == 0: # 沒path了 換下一個root吧
                print("-------")
                break

            new=np.empty( shape=(0,0))
            if l!=0 :  # 至多增加15個長度
                for j in range(Paths.shape[0]):
                    Path = Paths[j].reshape(1,-1)                   # print(Path[0])
                    pathArgNa = np.where( Path[0] < 0 )[0]          # print("pathArgNa" + str(pathArgNa))   打印出來是 [3] 這樣
                    Parity_computed = -1 * np.ones((1,8),dtype=int)
                    
                    if l >= 4: 
                        Parity_computed, _ = slow_compute_permissible_parity(Path, cs_decoded_tx_message, J, parityDistribution, l, useWhichMatrix, {})
                    
                    for k in range(listSize):
                        if l < 4:
                            new = np.vstack((new,np.hstack((Path.reshape(1,-1),np.array([[k]]))))) if new.size else np.hstack((Path.reshape(1,-1),np.array([[k]])))
                        else :  # now l >= 4:
                            index = slow_parity_check(Parity_computed, Path, k, cs_decoded_tx_message, J, messageLengthVector, parityDistribution, useWhichMatrix) 
                            if index:
                                new = np.vstack((new,np.hstack((Path.reshape(1,-1),np.array([[k]]))))) if new.size else np.hstack((Path.reshape(1,-1),np.array([[k]])))
                    
                    if len(pathArgNa) == 0:
                        new = np.vstack((new,np.hstack((Path.reshape(1,-1),np.array([[-1]]))))) if new.size else np.hstack((Path.reshape(1,-1),np.array([[-1]])))

                Paths = new 
            
            else: # 不再增加長度了
                # print("!!!!開始檢查完整的path們!!!")
                PathsUpdated = np.empty( shape=(0,0))
                for j in range(Paths.shape[0]):
                    # print("****第" + str(j) +"條path, 檢查開始")
                    isOkay = False
                    Path = Paths[j].reshape(1,-1)
                    isOkay = slow_parity_check(_, Path,_, cs_decoded_tx_message,J,messageLengthVector, parityDistribution, useWhichMatrix)
                    if isOkay:
                        PathsUpdated = np.vstack((PathsUpdated, Path)) if PathsUpdated.size else Path
                Paths = PathsUpdated



        if Paths.shape[0] >= 1: # 我們在這裡容許所有 有一個 outage的
            print("有candidate！！")
            optimalOne = 0
            if Paths.shape[0] >= 2:
                pathVar = np.zeros((Paths.shape[0]))
                for whichPath in range(Paths.shape[0]):
                    fadingValues = []
                    for l in range(Paths.shape[1]): 
                        if Paths[whichPath][l] != -1:
                            fadingValues.append( decBetaNoised[ Paths[whichPath][l] ][l] )
                    pathVar[whichPath] = np.var(fadingValues)
                optimalOne = np.argmin(pathVar)

            onlyPathToConsider = Paths[optimalOne]
            sectionLost = np.where(onlyPathToConsider < 0)[0]
            decoded_message = np.zeros((1, L*J), dtype=int)
            for l in np.arange(L):
                if (l not in sectionLost):
                    decoded_message[0, l*J:(l+1)*J] = cs_decoded_tx_message[onlyPathToConsider[l], l*J:(l+1)*J]

            recovered_message = slow_recover_msg(sectionLost, decoded_message, parityDistribution, messageLengthVector, J, L, useWhichMatrix)
            if recovered_message != np.array([], dtype= int).reshape(1,-1):
                tree_decoded_tx_message = np.vstack((tree_decoded_tx_message, recovered_message)) if tree_decoded_tx_message.size else recovered_message

    return tree_decoded_tx_message
