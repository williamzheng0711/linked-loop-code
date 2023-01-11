import numpy as np


######### The following are not used for linked-loop code
######### Hence pls feel free to just ignore them

def generate_parity_distribution(identity=False):
    parityDistribution = np.zeros((16,16),dtype=int)
    parityDistribution[0][1] = 7; parityDistribution[0][2] = 4; parityDistribution[0][3] = 3; parityDistribution[0][4] = 2; 
    parityDistribution[1][2] = 3; parityDistribution[1][3] = 2; parityDistribution[1][4] = 2; parityDistribution[1][5] = 2; 
    parityDistribution[2][3] = 3; parityDistribution[2][4] = 2; parityDistribution[2][5] = 2; parityDistribution[2][6] = 2; 
    
    for i in np.arange(3,12,1):
        for j in np.arange(i + 1, i + 5, 1):
            parityDistribution[i][j] = 2

    parityDistribution[12][13] = 3; parityDistribution[12][14] = 2; parityDistribution[12][15] = 3
    parityDistribution[13][14] = 3; parityDistribution[13][15] = 4;  
    parityDistribution[14][15] = 7

    if identity!= True:
        useWhichMatrix = np.zeros((16,16),dtype=int)
        for row in np.arange(0,16):
            for col in np.arange(0, 16):
                if parityDistribution[row][col]!=0:
                    dim = parityDistribution[row][col]
                    choices = matrix_repo(dim=dim)
                    # print(choices)
                    useWhichMatrix[row][col] = np.random.randint(low=0, high=len(choices))

    elif identity == True:
        useWhichMatrix = np.zeros((16,16),dtype=int)
        for row in np.arange(0,16):
            for col in np.arange(0, 16):
                if parityDistribution[row][col]!=0:
                    dim = parityDistribution[row][col]
                    choices = matrix_repo(dim=dim)
                    # print(choices)
                    useWhichMatrix[row][col] = 0

    return parityDistribution, useWhichMatrix




def Tree_decoder_uninformative_fading_growinglS(decBetaSignificants, decBetaSignificantsPos, G,L,J,B,parityLengthVector,messageLengthvector,listSize):
    # decBetaSignificants size is (16, "listSize")
    # Now "decBetaSignificants" are dictionary, not ordinary array
    cs_decoded_tx_message = np.ones( (listSize[-1], L*J) ) # (listSize[-1], 256)
    cs_decoded_tx_message = -1 * cs_decoded_tx_message

    for idx_l in range(L):
        for idx_ls in range(listSize[idx_l]):
            a = np.binary_repr(decBetaSignificantsPos[idx_l][idx_ls], width=J)
            # print("a = " + str(a))
            b = np.array([int(n) for n in a] ).reshape(1,-1)
            # print("b = " + str(b))
            cs_decoded_tx_message[idx_ls,   idx_l*J: (idx_l+1)*J ] = b[0, 0:J]

    listSizeOrder = np.argsort( decBetaSignificants[0] )
    # print("listSizeOrder is " + str(listSizeOrder))

    tree_decoded_tx_message = np.empty(shape=(0,0))
    for i in listSizeOrder:
        Paths = np.array([[i]])
        for l in range(1,L):
            # Grab the parity generator matrix corresponding to this section
            G1 = G[l-1]
            new=np.empty( shape=(0,0))
            for j in range(Paths.shape[0]):
                Path=Paths[j].reshape(1,-1)
                # print("i=" + str(i) + " j=" + str(j) + " and Path is" + str(Path))
                # Compute the permissible parity check bits for the section
                Parity_computed = compute_permissible_parity(Path,cs_decoded_tx_message,G1,L,J,parityLengthVector,messageLengthvector)
                # print("Parity_computed is: " + str(Parity_computed) )
                for k in range(listSize[l]):
                    # Verify parity constraints for the children of surviving path
                    index = parity_check(Parity_computed,Path,k,cs_decoded_tx_message,L,J,parityLengthVector,messageLengthvector)
                    # If parity constraints are satisfied, update the path
                    if index:
                        new = np.vstack((new,np.hstack((Path.reshape(1,-1),np.array([[k]]))))) if new.size else np.hstack((Path.reshape(1,-1),np.array([[k]])))
            Paths = new 
            
        if Paths.shape[0] >= 2:
            # If tree decoder outputs multiple paths for a root node, select the first one 
            flag = check_if_identical_msgs(Paths, cs_decoded_tx_message, L,J,parityLengthVector,messageLengthvector)
            if flag:
                # print("Path[0] detail is " + str(Paths[0]))
                tree_decoded_tx_message = np.vstack((tree_decoded_tx_message,extract_msg_bits(Paths[0].reshape(1,-1),cs_decoded_tx_message, L,J,parityLengthVector,messageLengthvector))) if tree_decoded_tx_message.size else extract_msg_bits(Paths[0].reshape(1,-1),cs_decoded_tx_message, L,J,parityLengthVector,messageLengthvector)
            else:
                # print("Path shape is" + str(Paths.shape))
                # print("Path[0] detail is " + str(Paths[0]))
                optimalOne = 0
                pathVar = np.zeros((Paths.shape[0]))
                for whichPath in range(Paths.shape[0]):
                    fadingValues = []
                    for l in range(Paths.shape[1]):
                        # decBetaSignificantsPos size is (16, "listSize")
                        fadingValues.append( decBetaSignificants[l][ Paths[whichPath][l] ] )
                    
                    pathVar[whichPath] = np.var(fadingValues)

                optimalOne = np.argmin(pathVar)
                tree_decoded_tx_message = np.vstack((tree_decoded_tx_message,extract_msg_bits(Paths[optimalOne].reshape(1,-1),cs_decoded_tx_message, L,J,parityLengthVector,messageLengthvector))) if tree_decoded_tx_message.size else extract_msg_bits(Paths[optimalOne].reshape(1,-1),cs_decoded_tx_message, L,J,parityLengthVector,messageLengthvector)
                # tree_decoded_tx_message = np.vstack( (tree_decoded_tx_message,extract_msg_bits(Paths.reshape(Paths.shape[0],-1),cs_decoded_tx_message, L,J,parityLengthVector,messageLengthvector))) if tree_decoded_tx_message.size else extract_msg_bits(Paths.reshape(Paths.shape[0],-1),cs_decoded_tx_message, L,J,parityLengthVector,messageLengthvector)
        elif Paths.shape[0] == 1:
            tree_decoded_tx_message = np.vstack((tree_decoded_tx_message,extract_msg_bits(Paths.reshape(1,-1),cs_decoded_tx_message, L,J,parityLengthVector,messageLengthvector))) if tree_decoded_tx_message.size else extract_msg_bits(Paths.reshape(1,-1),cs_decoded_tx_message, L,J,parityLengthVector,messageLengthvector)
    
    return tree_decoded_tx_message




def Tree_decoder_uninformative_fading(decBetaSignificants, decBetaSignificantsPos, G,L,J,B,parityLengthVector,messageLengthvector,listSize):
    # decBetaSignificants size is (listSize, 16)
    cs_decoded_tx_message = np.zeros( (listSize, L*J) ) # (listSize, 256)
    for id_row in range(decBetaSignificantsPos.shape[0]):
        for id_col in range(decBetaSignificantsPos.shape[1]):
            a = np.binary_repr(decBetaSignificantsPos[id_row][id_col], width=J)
            # print("a = " + str(a))
            b = np.array([int(n) for n in a] ).reshape(1,-1)
            # print("b = " + str(b))
            cs_decoded_tx_message[ id_row, id_col*J: (id_col+1)*J ] = b[0, 0:J]

    listSizeOrder = np.argsort( decBetaSignificants[:,0] )
    # print("listSizeOrder is " + str(listSizeOrder))

    tree_decoded_tx_message = np.empty(shape=(0,0))
    for i in listSizeOrder:
        Paths = np.array([[i]])
        for l in range(1,L):
            # Grab the parity generator matrix corresponding to this section
            G1 = G[l-1]
            new=np.empty( shape=(0,0))
            for j in range(Paths.shape[0]):
                Path=Paths[j].reshape(1,-1)
                # print("i=" + str(i) + " j=" + str(j) + " and Path is" + str(Path))
                # Compute the permissible parity check bits for the section
                Parity_computed = compute_permissible_parity(Path,cs_decoded_tx_message,G1,L,J,parityLengthVector,messageLengthvector)
                # print("Parity_computed is: " + str(Parity_computed) )
                for k in range(listSize):
                    # Verify parity constraints for the children of surviving path
                    index = parity_check(Parity_computed,Path,k,cs_decoded_tx_message,L,J,parityLengthVector,messageLengthvector)
                    # If parity constraints are satisfied, update the path
                    if index:
                        new = np.vstack((new,np.hstack((Path.reshape(1,-1),np.array([[k]]))))) if new.size else np.hstack((Path.reshape(1,-1),np.array([[k]])))
            Paths = new 
            
        if Paths.shape[0] >= 2:
            # If tree decoder outputs multiple paths for a root node, select the first one 
            flag = check_if_identical_msgs(Paths, cs_decoded_tx_message, L,J,parityLengthVector,messageLengthvector)
            if flag:
                # print("Path[0] detail is " + str(Paths[0]))
                tree_decoded_tx_message = np.vstack((tree_decoded_tx_message,extract_msg_bits(Paths[0].reshape(1,-1),cs_decoded_tx_message, L,J,parityLengthVector,messageLengthvector))) if tree_decoded_tx_message.size else extract_msg_bits(Paths[0].reshape(1,-1),cs_decoded_tx_message, L,J,parityLengthVector,messageLengthvector)
            else:
                # print("Path shape is" + str(Paths.shape))
                # print("Path[0] detail is " + str(Paths[0]))
                optimalOne = 0
                pathVar = np.zeros((Paths.shape[0]))
                for whichPath in range(Paths.shape[0]):
                    fadingValues = []
                    for l in range(Paths.shape[1]):
                        # decBetaSignificantsPos size is (listSize, 16)s
                        fadingValues.append( decBetaSignificants[ Paths[whichPath][l] ][l] )
                    
                    # print("fadingValues = " + str(fadingValues))
                    demeanFading = fadingValues - np.mean(fadingValues)
                    # pathVar[whichPath] = np.linalg.norm(demeanFading, 1)
                    pathVar[whichPath] = np.var(fadingValues)

                optimalOne = np.argmin(pathVar)
                tree_decoded_tx_message = np.vstack((tree_decoded_tx_message,extract_msg_bits(Paths[optimalOne].reshape(1,-1),cs_decoded_tx_message, L,J,parityLengthVector,messageLengthvector))) if tree_decoded_tx_message.size else extract_msg_bits(Paths[optimalOne].reshape(1,-1),cs_decoded_tx_message, L,J,parityLengthVector,messageLengthvector)
                # tree_decoded_tx_message = np.vstack( (tree_decoded_tx_message,extract_msg_bits(Paths.reshape(Paths.shape[0],-1),cs_decoded_tx_message, L,J,parityLengthVector,messageLengthvector))) if tree_decoded_tx_message.size else extract_msg_bits(Paths.reshape(Paths.shape[0],-1),cs_decoded_tx_message, L,J,parityLengthVector,messageLengthvector)
        elif Paths.shape[0] == 1:
            tree_decoded_tx_message = np.vstack((tree_decoded_tx_message,extract_msg_bits(Paths.reshape(1,-1),cs_decoded_tx_message, L,J,parityLengthVector,messageLengthvector))) if tree_decoded_tx_message.size else extract_msg_bits(Paths.reshape(1,-1),cs_decoded_tx_message, L,J,parityLengthVector,messageLengthvector)
    
    return tree_decoded_tx_message





def postprocess(β1_dec, L, J, listSize):
    toGiveOut = np.array([])
    for i in range(L):
        A = β1_dec[i*2**J:(i+1)*2**J]
        idx = (A.reshape(2**J,)).argsort()[np.arange(2**J-listSize)]
        B = np.setdiff1d(np.arange(2**J),idx)
        temp = np.zeros(2**J)
        temp[B] = 1
        temp = temp.reshape(-1)

        toGiveOut = np.concatenate((toGiveOut,temp))
    return np.array(toGiveOut)



def postprocess_increasinglS(β1_dec, L, J, listSize):
    toGiveOut = np.array([])
    for i in range(L):
        # extract the information of i-th section
        A = β1_dec[i*2**J:(i+1)*2**J]
        # idx is the indices of the smallest (2**J - listSize) entries (fading values) in section i.
        idx = (A.reshape(2**J,)).argsort()[np.arange(2**J-listSize[i], dtype=int)]
        # B = np.setdiff1d(np.arange(2**J),idx)
        # temp = np.zeros(2**J)
        temp = A
        # idx是小值indices 所以都變成0
        temp[idx] = 0
        temp = temp.reshape(-1)

        toGiveOut = np.concatenate((toGiveOut,temp))
    return np.array(toGiveOut)


# given some decBeta, this algorithm gives out a most probable outer code
def extract_one_outer_code_listSize(decBetaSignificants, decBetaSignificantsPos, K, G, J, P, Ml, messageLengthVector, parityLengthVector):
    usedUp = False 
    fadingValues = [] # entries are in range (0, infty)
    positionValues = [] # entries are in range (0, 65535) = (0, 2**J-1)

    L = decBetaSignificants.shape[0] # usually, L = 16
    # print("L = " + str(L))
    eachSectionArgs = np.zeros(L, dtype=int)

    while len(fadingValues) < L and usedUp == False: 
        # print("l = " + str(l))
        # print("positionValues = " + str(positionValues))

        lenFVsPrior = len(fadingValues)

        eachSectionArgs[ lenFVsPrior + 1 : ] = 0

        if (len(fadingValues)!=0):
            # 找最小的
            args = np.argsort( abs(decBetaSignificants[lenFVsPrior] - np.mean(fadingValues)) )
        else: 
            # 找最大的
            args = np.flip( np.argsort( decBetaSignificants[lenFVsPrior] ) )
        
        # if (eachSectionArgs[ len(fadingValues) ] == 0):
        #     arg_trials = eachSectionArgs[ len(fadingValues) ] 
        # else: 
        #     arg_trials = eachSectionArgs[ len(fadingValues) ] + 1

        arg_trials = eachSectionArgs[ len(fadingValues) ]

        parityConsistent = False  
        
        while parityConsistent == False and arg_trials < len(args) and usedUp == False:
            positionValues.append( decBetaSignificantsPos[ lenFVsPrior ][args[arg_trials]]   )
            fadingValues.append(      decBetaSignificants[ lenFVsPrior ][args[arg_trials]]   )
            if ( len(positionValues) >= 2 ):
                if positionValues[-1] >= 0:
                    parityConsistent = parity_check_part(K, G, L, J, P, Ml, messageLengthVector, parityLengthVector, positionValues)
                
                if (positionValues[-1] < 0 or parityConsistent == False):
                    positionValues.pop()
                    fadingValues.pop()
            else: # is len(positionValues) == 1, then true
                parityConsistent = True
            
            arg_trials += 1

        if len(fadingValues) == lenFVsPrior and len(fadingValues) != 0:
            # 我們失敗了, we retreat to last section
            positionValues.pop()
            fadingValues.pop()

        # elif len(fadingValues) == lenFVsPrior and len(fadingValues) == 0:
        #     usedUp = True
        #     break

        elif len(fadingValues) == lenFVsPrior and len(fadingValues) == 0:
            decBetaSignificants[   0][args[0]] = 0
            decBetaSignificantsPos[0][args[0]] = -1

        elif len(fadingValues) == lenFVsPrior + 1:
            # 成功, we succeed to find one proper candidate in this section
            eachSectionArgs[lenFVsPrior] = arg_trials

        if max(decBetaSignificants[   0]) == 0:
            usedUp == True
            break
        
    # anOuterCode = convert_positions_to_bits(positionValues)
    if usedUp == False and len(positionValues)==L:
        # for ll in np.arange(L):
        # Why in original code we only erase the root? 
        for ll in np.arange(0,1):
            index_l = np.where(decBetaSignificantsPos[ll] == positionValues[ll])
            # print("index of " + str(ll) + " is: " + str(index_l))
            decBetaSignificants[ll][index_l] = 0
            decBetaSignificantsPos[ll][index_l] = -1
        
        # print("fadingValues = " + str(fadingValues))
        # print("positionValues = " + str(positionValues))
        # print("eachSectionArgs = " + str(eachSectionArgs))

        return positionValues, usedUp, decBetaSignificants, decBetaSignificantsPos
    
    else: 
        return [], usedUp, decBetaSignificants, decBetaSignificantsPos
        

# 原先 tx_message 是 100 x 128的東西
def parity_check_part(K, G, L, J, P, Ml, messageLengthVector, parityLengthVector, positionValues):
    
    howManySectionInHand = len(positionValues)
    # print("Now howmany sections? " + str(howManySectionInHand))
    
    answer = np.array([])
    for l in np.arange( howManySectionInHand ):
        # print(positionValues)
        a = np.binary_repr(positionValues[l], width= J )
        answer = np.append(answer, [int(n) for n in a] ).reshape(1,-1)

    answer = np.array(answer).reshape(1,-1)
    # print("answer :" + str(answer))

    messageBySections = np.array([])
    # section ZERO is always okay to go
    currentSection = 0
    while (currentSection < howManySectionInHand ):
        messageBySections = np.append(messageBySections, answer[:, currentSection*messageLengthVector[0] : (currentSection+1)*messageLengthVector[0]-parityLengthVector[currentSection]])
        currentSection += 1

    messageBySections = messageBySections.reshape(1,-1)
    # print( "messageBySections :" + str(messageBySections) )
    
    # for i in range(1,L):
    i = howManySectionInHand - 1
    ParityInteger=np.zeros((1,1),dtype='int')
    G1=G[i-1]
    for j in range(1,i+1):
        ParityBinary = np.mod(
                            np.matmul(  messageBySections[ 0,np.sum(messageLengthVector[0:j-1]) : np.sum(messageLengthVector[0:j]) ],
                                        G1[ np.sum(messageLengthVector[0:j-1]) : np.sum(messageLengthVector[0:j]) ]
                                ),
                        2)
        # Convert into decimal equivalent\n",
        ParityBinary = ParityBinary.reshape(1,-1)
        # print("ParityBinary shape: " + str(ParityBinary.shape))
        ParityInteger1 = ParityBinary.dot(2**np.arange(ParityBinary.shape[1])[::-1]).reshape([1,1])
        ParityInteger = np.mod(ParityInteger+ParityInteger1,2**parityLengthVector[i])


    Parity = np.array([list(np.binary_repr(int(x),parityLengthVector[i])) for x in ParityInteger], dtype=int)
    # print("Parity via Calculation is = " + str(Parity))
    onHand = np.array(answer[:, answer.shape[1] - parityLengthVector[howManySectionInHand-1] : ], dtype=int).reshape(1,-1)
    # print("On hand is = " + str(onHand) )

    if np.array_equal(Parity, onHand):
        same = True
    else:
        same = False
    
    # print("checked " + str(same))

    return same


def stitching_use_fading_and_tree_listSize( decBetaSignificants, decBetaSignificantsPos, K, G, J, P, Ml, messageLengthVector, parityLengthVector, L, listSize ):
    # shape of decBeta is 16 * 65536 ( L*2**J)

    rxOutercodes = np.array([])
    nowRecovered = 0
    alarmCondition = False
    decBetaSignificantsUpdate = decBetaSignificants
    decBetaSignificantsPosUpdate = decBetaSignificantsPos

    while (nowRecovered < K and alarmCondition == False):

        print("nowRecovered is " + str(nowRecovered))
        anOuterCode, _, decBetaSignificantsUpdate, decBetaSignificantsPosUpdate = extract_one_outer_code_listSize(decBetaSignificantsUpdate, decBetaSignificantsPosUpdate, K, G, J, P, Ml, messageLengthVector, parityLengthVector)
        if len(anOuterCode) != 0:
            answer = np.array([])
            for l in np.arange(L):
                a = np.binary_repr(anOuterCode[l], width=16)
                # answer.append( [int(n) for n in a] )
                answer = np.append(answer, [int(n) for n in a] ).reshape(1,-1)

            if nowRecovered == 0:
                rxOutercodes = answer
            else: 
                rxOutercodes = np.vstack((rxOutercodes, answer)) 
            nowRecovered += 1
        else: # if usedUp is true, then anOuterCode is null
            alarmCondition = True
    
    # rxOutercodes 最好是 (100, 256)的binary 可以直接進行 PUPE 的計算
    return rxOutercodes


def Tree_decoder_uninformative(cs_decoded_tx_message,G,L,J,B,parityLengthVector,messageLengthvector,listSize):
    tree_decoded_tx_message = np.empty(shape=(0,0))
    for i in range(listSize):
        Paths = np.array([[i]])
        for l in range(1,L):
            # Grab the parity generator matrix corresponding to this section
            G1 = G[l-1]
            new=np.empty( shape=(0,0))
            for j in range(Paths.shape[0]):
                Path=Paths[j].reshape(1,-1)
                # print("i=" + str(i) + " j=" + str(j) + " and Path is" + str(Path))
                # Compute the permissible parity check bits for the section
                Parity_computed = compute_permissible_parity(Path,cs_decoded_tx_message,G1,L,J,parityLengthVector,messageLengthvector)
                # print("Parity_computed is: " + str(Parity_computed) )
                for k in range(listSize):
                    # Verify parity constraints for the children of surviving path
                    index = parity_check(Parity_computed,Path,k,cs_decoded_tx_message,L,J,parityLengthVector,messageLengthvector)
                    # If parity constraints are satisfied, update the path
                    if index:
                        new = np.vstack((new,np.hstack((Path.reshape(1,-1),np.array([[k]]))))) if new.size else np.hstack((Path.reshape(1,-1),np.array([[k]])))
            Paths = new 
            
        if Paths.shape[0] >= 2:
            # If tree decoder outputs multiple paths for a root node, select the first one 
            flag = check_if_identical_msgs(Paths, cs_decoded_tx_message, L,J,parityLengthVector,messageLengthvector)
            if flag:
                tree_decoded_tx_message = np.vstack((tree_decoded_tx_message,extract_msg_bits(Paths[0].reshape(1,-1),cs_decoded_tx_message, L,J,parityLengthVector,messageLengthvector))) if tree_decoded_tx_message.size else extract_msg_bits(Paths[0].reshape(1,-1),cs_decoded_tx_message, L,J,parityLengthVector,messageLengthvector)
            else:
                # print("Path shape is" + str(Paths.shape))
                tree_decoded_tx_message = np.vstack((tree_decoded_tx_message,extract_msg_bits(Paths.reshape(Paths.shape[0],-1),cs_decoded_tx_message, L,J,parityLengthVector,messageLengthvector))) if tree_decoded_tx_message.size else extract_msg_bits(Paths.reshape(Paths.shape[0],-1),cs_decoded_tx_message, L,J,parityLengthVector,messageLengthvector)
        elif Paths.shape[0] == 1:
            tree_decoded_tx_message = np.vstack((tree_decoded_tx_message,extract_msg_bits(Paths.reshape(1,-1),cs_decoded_tx_message, L,J,parityLengthVector,messageLengthvector))) if tree_decoded_tx_message.size else extract_msg_bits(Paths.reshape(1,-1),cs_decoded_tx_message, L,J,parityLengthVector,messageLengthvector)
    
    return tree_decoded_tx_message



def amp_prior_art(y, σ_n, P, L, M, T, Ab, Az, p0, K):
    n = y.size
    β = np.zeros((L*M, 1))
    z = y
    Phat = n*P/L
    
    for t in range(T):
        
        τ = np.sqrt(np.sum(z**2)/n)
        # effective observation
        s = (np.sqrt(Phat)*β + Az(z)).astype(np.longdouble) 
        # denoiser
        β = (p0*np.exp(-(s-np.sqrt(Phat))**2/(2*τ**2)))/ (p0*np.exp(-(s-np.sqrt(Phat))**2/(2*τ**2)) + (1-p0)*np.exp(-s**2/(2*τ**2))).astype(float).reshape(-1, 1)
        # residual
        z = y - np.sqrt(Phat)*Ab(β) + (z/(n*τ**2)) * (Phat*np.sum(β) - Phat*np.sum(β**2))
        #print(t,τ)

    return β




def amp_prior_art_Rician(y, σ_n, P, L, M, T, Ab, Az, p0, K, v_Rician, sigma_Rician, convertToBeta):
    n = y.size
    β = np.zeros((L*M, 1))
    z = y
    Phat = n*P/L
    dl = np.sqrt(Phat)
    print("dl=" +str(dl))
    
    for t in range(T):
        print("-------------------Iter "+str(t) +" begins-------------------")

        # estimated s.e. of ζ
        τ = np.sqrt(np.sum(z**2)/n)

        # effective observation     r = d * hs + τζ
        r = (np.sqrt(Phat)*β + Az(z)).astype(np.longdouble) 
        print("r length is " + str(len(r)))

        print("r[0]=" + str(r[0]))
        print("tau=" + str(τ))

        # print(str(normal))
        # print(str(rice))

        # denoiser: β is E[hs|r]
        # if (t < T-1):
        # Rician distribution pdf: rice.pdf(h, v_Rician/sigma_Rician, 0, sigma_Rician) =  h/sigma_Rician**2 * np.exp(-(h**2 + v_Rician**2)/(2*sigma_Rician**2)) * i0(v_Rician/sigma_Rician**2 * h) 
        
        start = timeit.default_timer()
        Nume_int_part, _ = quad_vec(f = lambda h:   h*normal.pdf((r-dl*h)/τ) * rice.pdf(h, v_Rician/sigma_Rician, 0, sigma_Rician), a=0, b=np.Infinity) 
        stop = timeit.default_timer()
        print('Nume okay. Time: ' + str(stop - start))     

        start = timeit.default_timer()
        Deno_int_part, _ = quad_vec(f = lambda h:     normal.pdf((r-dl*h)/τ) * rice.pdf(h, v_Rician/sigma_Rician, 0, sigma_Rician), a=0, b=np.Infinity) 
        stop = timeit.default_timer()
        print('Deno okay. Time: ' + str(stop - start)) 

        # β is E[hs|r]
        β = p0 * Nume_int_part / ( p0 * Deno_int_part  +   (1-p0) * normal.pdf(r/τ) )                                     
        print("sum(beta) is: " + str(sum(β)))
    
        # residual
        # z = y - np.sqrt(Phat)*Ab(β) + (z/(n*τ**2)) * (Phat*np.sum(β) - Phat*np.sum((β)**2))
        int_part, _ =  quad_vec(f = lambda h:   h**2 * normal.pdf((r-dl*h)/τ) * rice.pdf(h, v_Rician/sigma_Rician, 0, sigma_Rician), a=0, b=np.Infinity) 
        # E_h2s2 = E[h^2 s^2 | r]
        E_h2s2 = p0 * int_part / ( p0 * Deno_int_part  +   (1-p0) * normal.pdf(r/τ) )     

        z = y - np.sqrt(Phat)*Ab(β) + (z/n) * (Phat/τ**2) * sum(  E_h2s2 - β**2  )


    if convertToBeta:
        pme_int_part, _ = quad_vec(f = lambda h:     normal.pdf((r-dl*h)/τ) *  rice.pdf(h, v_Rician/sigma_Rician, 0, sigma_Rician)      , a=0, b=np.Infinity,) 
        β = p0 * pme_int_part / ( p0 * pme_int_part + (1-p0) * normal.pdf(r/τ) )
    print(β)

    return β


# P : Total number of parity check bits
# Ml: Total number of information bits
def Tree_error_correct_encode(tx_message,K,L,J,P,Ml,messageLengthVector,parityLengthVector, parityDistribution, useWhichMatrix):
    encoded_tx_message = np.zeros((K,Ml+P),dtype=int)
    # plug in the info bits for each section
    encoded_tx_message[:,0:messageLengthVector[0]] = tx_message[:,0:messageLengthVector[0]]
    for i in range(1,L):
        encoded_tx_message[:,i*J:i*J+messageLengthVector[i]] = tx_message[:,np.sum(messageLengthVector[0:i]):np.sum(messageLengthVector[0:i+1])]
    
    for i in np.arange(0,L,1):
        parityDistRow_i = np.nonzero(parityDistribution[i])[0]
        # print(parityDistRow_i)
        for j in parityDistRow_i:
            # when i=0, j will be 1 2 3 4
            # j 是要写入东西的section i 是东西的来源section
            # if (i==0 and j == 1):
            # print("---")
            # print(j*J+messageLengthVector[j]+sum(parityDistribution[0:i,j]))
            # print(j*J+messageLengthVector[j]+sum(parityDistribution[0:i+1,j]))
            # print( sum(messageLengthVector[0:i])+sum(parityDistribution[i,0:j]) )
            # print( sum(messageLengthVector[0:i])+sum(parityDistribution[i,0:j+1]))
            # print(i,j)
            if useWhichMatrix != [] :
                encoded_tx_message[:,j*J+messageLengthVector[j]+sum(parityDistribution[0:i,j]) :    j*J+messageLengthVector[j]+sum(parityDistribution[0:i+1,j])] = \
                    (np.matmul(np.array(matrix_repo(parityDistribution[i][j])[useWhichMatrix[i][j]]),(tx_message[:, sum(messageLengthVector[0:i])+sum(parityDistribution[i,0:j]) : sum(messageLengthVector[0:i])+sum(parityDistribution[i,0:j+1])]).transpose() )).transpose() % 2

            else: 
                encoded_tx_message[:,j*J+messageLengthVector[j]+sum(parityDistribution[0:i,j]) :    j*J+messageLengthVector[j]+sum(parityDistribution[0:i+1,j])] = \
                    tx_message[:, sum(messageLengthVector[0:i])+sum(parityDistribution[i,0:j]) : sum(messageLengthVector[0:i])+sum(parityDistribution[i,0:j+1])]


    np.savetxt('encoded_message.csv', encoded_tx_message[0].reshape(16,16), delimiter=',', fmt='%d')
    # print(encoded_tx_message[0,0:16])

    return encoded_tx_message


def convert_bits_to_sparse(encoded_tx_message,L,J,K):
    encoded_tx_message_sparse=np.zeros((L*2**J,1),dtype=float)
    for i in range(L):
        A = encoded_tx_message[:,i*J:(i+1)*J]
        B = A.dot(2**np.arange(J)[::-1]).reshape([K,1])
        np.add.at(encoded_tx_message_sparse, i*2**J+B, 1)        
    return encoded_tx_message_sparse

def convert_bits_to_sparse_Rician(encoded_tx_message,L,J,K, v_Rician, sigma_Rician):
    count = 0
    encoded_tx_message_sparse=np.zeros((L*2**J,1),dtype=float)
    fading_coefficients = rice.rvs(b= v_Rician/sigma_Rician, scale= sigma_Rician, loc =0, size=K)
    print(fading_coefficients)
    for i in range(L):
        A = encoded_tx_message[:,i*J:(i+1)*J]
        B = A.dot(2**np.arange(J)[::-1]).reshape([K,1])
        for k in range(K):
            encoded_tx_message_sparse[i*2**J+B[k]] += fading_coefficients[k]
            # np.add.at(encoded_tx_message_sparse, i*2**J+B[k], fading_coefficients[k])        
            count += 1
    
    return encoded_tx_message_sparse



def generate_parity_matrix(L,messageLengthVector,parityLengthVector):
    # Generate a full matrix, use only the portion needed for tree code
    G = []
    for i in range(1,L):
        Gp = np.random.randint(2,size=(np.sum(messageLengthVector[0:i]),parityLengthVector[i])).tolist()
        G.append(Gp)
    # return np.asarray(G,)
    return np.asarray(G, dtype=object)


def Tree_error_correct_encode_tb(tx_message,K,L,J,P,Ml,messageLengthVector,parityLengthVector, parityDistribution):
    encoded_tx_message = np.zeros((K,Ml+P),dtype=int)
    # plug in the info bits for each section
    encoded_tx_message[:,0:messageLengthVector[0]] = tx_message[:,0:messageLengthVector[0]]
    for i in range(1,L):
        encoded_tx_message[:,i*J:i*J+messageLengthVector[i]] = tx_message[:,np.sum(messageLengthVector[0:i]):np.sum(messageLengthVector[0:i+1])]
    
    for i in np.arange(0,L,1):
        parityDistRow_i = np.nonzero(parityDistribution[i])[0]
        # print(parityDistRow_i)
        for j in parityDistRow_i:
            # when i=0, j will be 1 2 3 4
            # j 是要写入东西的section i 是东西的来源section
            # if (i==0 and j == 1):
            # print("---")
            # print(j*J+messageLengthVector[j]+sum(parityDistribution[0:i,j]))
            # print(j*J+messageLengthVector[j]+sum(parityDistribution[0:i+1,j]))
            # print( sum(messageLengthVector[0:i])+sum(parityDistribution[i,0:j]) )
            # print( sum(messageLengthVector[0:i])+sum(parityDistribution[i,0:j+1]))
            # print(i,j)
            encoded_tx_message[:,j*J+messageLengthVector[j]+sum(parityDistribution[0:i,j]) :    j*J+messageLengthVector[j]+sum(parityDistribution[0:i+1,j])] = \
                tx_message[:, sum(messageLengthVector[0:i])+sum(parityDistribution[i,0:j]) : sum(messageLengthVector[0:i])+sum(parityDistribution[i,0:j+1])]

    np.savetxt('encoded_message.csv', encoded_tx_message[0].reshape(16,16), delimiter=',', fmt='%d')
    return encoded_tx_message



def compute_permissible_parity(Path,cs_decoded_tx_message,G1,L,J,parityLengthVector,messageLengthvector):
    msg_bits = extract_msg_bits(Path,cs_decoded_tx_message, L,J,parityLengthVector,messageLengthvector)
    Lpath = Path.shape[1]
    Parity_computed_integer = 0
    for i in range(Lpath):
        ParityBinary = np.mod(np.matmul(msg_bits[:,np.sum(messageLengthvector[0:i]):np.sum(messageLengthvector[0:i+1])],
                            G1[np.sum(messageLengthvector[0:i]):np.sum(messageLengthvector[0:i+1])]),2)
        ParityBinary=ParityBinary.reshape(1,-1)
        # Convert into decimal equivalent\n",
        ParityInteger1 = ParityBinary.dot(2**np.arange(ParityBinary.shape[1])[::-1])
        Parity_computed_integer = np.mod(Parity_computed_integer+ParityInteger1,2**parityLengthVector[Lpath])        
         
    Parity_computed = np.array([list(np.binary_repr(int(x),parityLengthVector[Lpath])) for x in Parity_computed_integer], dtype=int)
    return Parity_computed



def parity_check(Parity_computed,Path,k,cs_decoded_tx_message,L,J,parityLengthVector,messageLengthvector, parityDistribution, useWhichMatrix):
    index1 = 0
    index2 = 1
    Lpath = Path.shape[1]
    Parity = cs_decoded_tx_message[k,Lpath*J+messageLengthvector[Lpath]:(Lpath+1)*J]
    check_args = np.where(Parity_computed >=0)[0]
    if (np.sum(np.absolute(Parity_computed[check_args]-Parity[check_args])) == 0):
        index1 = 1
    
    if Lpath >= 13:
        cs_decoded_L_sections = np.ones((1,L*J), dtype=int)
        for ll in np.arange(Lpath):
            cs_decoded_L_sections[0][ll*J:(ll+1)*J] = cs_decoded_tx_message[Path[0][ll], ll*J:(ll+1)*J]

        for l in np.arange(12, Lpath):
            # check what sections are partly determined by l
            toCheckSections = np.nonzero(parityDistribution[l])[0] # is l = 13, then toCheckSections = [14, 15, 0, 1]
            # for each those sections, check if parity are same. 
            for section in toCheckSections: # section = 14, 15, 0, 1
                if section > l: continue # only section = 0, 1 will be executed
                
                # print('---- l='+str(l) +" and section=" + str(section) + "----")
                # print(section*J + messageLengthvector[section] + sum(parityDistribution[0:l,section]))
                # print(section*J + messageLengthvector[section] + sum(parityDistribution[0:l+1,section]))
                # print(l*J        + sum(parityDistribution[l,0:section]))
                # print(l*J        + sum(parityDistribution[l,0:section+1]))

                gen_mat = matrix_repo(parityDistribution[l][section])[useWhichMatrix[l][section]]
                # gen_binmat = BinMatrix(gen_mat)

                oldPart = cs_decoded_L_sections[0][section*J + messageLengthvector[section] + sum(parityDistribution[0:l,section]) : section*J + messageLengthvector[section] + sum(parityDistribution[0:l+1,section])].reshape(1,-1)[0]
                newPart = np.matmul(gen_mat ,cs_decoded_L_sections[0][      l*J                                + sum(parityDistribution[l,0:section]) :       l*J                                + sum(parityDistribution[l,0:section+1])] ).reshape(1,-1)[0]

                check_args_old = np.where(oldPart >=0)[0]
                # print("check_args_old" + str(check_args_old))
                checksum =  np.sum(np.absolute(  oldPart[check_args_old]-newPart[check_args_old]  ))
                # print("checksum = " + str(checksum))
                if  checksum != 0:
                    index2 = 0
                    return index2

    return index1 * index2