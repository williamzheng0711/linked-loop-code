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


def slow_parity_check(Parity_computed,Path,k,cs_decoded_tx_message,J,messageLengthvector):
    index1 = 0
    index2 = 1
    Lpath = Path.shape[1]
    Parity = cs_decoded_tx_message[k,Lpath*J+messageLengthvector[Lpath]:(Lpath+1)*J]
    if (np.sum(np.absolute(Parity_computed-Parity)) == 0):
        index1 = 1

    return index1 * index2




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
