import numpy as np
from utils import *

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
        parity_i = np.zeros((100, 8), dtype=int )
        for decider in whoDecidesI:
            parity_i = parity_i + np.matmul( info[decider], matrix_repo(dim=8)[useWhichMatrix[decider][i]] )
        encoded_tx_message[:, i*J+np.sum(messageLengthVector[i]) : (i+1)*J] = np.mod(parity_i, 2)

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

    for i, arg_i in zip(listSizeOrder, np.arange(len(listSizeOrder))):
        Paths = np.array([[i]])
        # not tail bit yet
        for l in range(1, L):
            # Grab the parity generator matrix corresponding to this section  
            new=np.empty( shape=(0,0))
            for j in range(Paths.shape[0]):
                Path=Paths[j].reshape(1,-1) 
                Parity_computed= np.ones((1,8),dtype=int) if l<4 else slow_compute_permissible_parity(Path, cs_decoded_tx_message, J, parityDistribution, l, useWhichMatrix)
                # print("Parity_computed is: " + str(Parity_computed) )
                for k in range(listSize):
                    # Verify parity constraints for the children of surviving path
                    index = l<4 or slow_parity_check(Parity_computed, Path, k, cs_decoded_tx_message, J, messageLengthVector)
                    # If parity constraints are satisfied, update the path
                    if index:
                        new = np.vstack((new,np.hstack((Path.reshape(1,-1),np.array([[k]]))))) if new.size else np.hstack((Path.reshape(1,-1),np.array([[k]])))
            Paths = new 
            print("l=" + str(l) + ' path number: ' + str(Paths.shape[0]))
        
        # let's do tail biting!!!
        print("跑到這了 有多少條path？" + str(Paths.shape[0]))
        PathsUpdated = np.empty( shape=(0,0))
        for j in range(Paths.shape[0]):
            isOkay = True
            Path = Paths[j].reshape(1,-1)
            for ll in [0,1,2,3]:
                Parity_computed_ll = slow_compute_permissible_parity(Path,cs_decoded_tx_message,J, parityDistribution, ll, useWhichMatrix=useWhichMatrix)
                flag_ll = sum(np.abs(Parity_computed_ll - cs_decoded_tx_message[Path[0][ll], ll*J:ll*J+8]))
                if flag_ll !=0: 
                    isOkay = False
                    break
            if isOkay:
                PathsUpdated = np.vstack(PathsUpdated, Path)
        Paths = PathsUpdated
        print("跑到這了!!!")


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
            print("有一條消息~")   
        
        print(str(arg_i) + " 又跑完了一個root")

    return tree_decoded_tx_message, usedRootsIndex


def slow_compute_permissible_parity(Path,cs_decoded_tx_message,J, parityDistribution, toCheck, useWhichMatrix):
    # if path length  = 2
    # then we wanna have parity for section 2. section2Check = 2
    section2Check = toCheck
    parityDist = parityDistribution[:,section2Check].reshape(1,-1)[0]
    # print("----------parityDist: " + str(parityDist))
    deciders = np.nonzero(parityDist)[0]
    # print("parityDependents = " + str(parityDependents))
    Parity_computed = np.zeros(8, dtype=int)

    focusPath = Path[0]
    for decider in deciders:      # l labels the sections we gonna check to fix section2Check's parities
        gen_mat = matrix_repo(dim=8)[useWhichMatrix[decider][section2Check]] 
        Parity_computed = Parity_computed + np.matmul( cs_decoded_tx_message[focusPath[decider], decider*J : decider*J+8], gen_mat)

    Parity_computed = np.mod(Parity_computed, 2)
    return Parity_computed



def slow_parity_check(Parity_computed,Path,k,cs_decoded_tx_message,J,messageLengthvector):
    index1 = 0
    index2 = 1
    Lpath = Path.shape[1]
    Parity = cs_decoded_tx_message[k,Lpath*J+messageLengthvector[Lpath]:(Lpath+1)*J]
    if (np.sum(np.absolute(Parity_computed-Parity)) == 0):
        index1 = 1

    return index1 * index2