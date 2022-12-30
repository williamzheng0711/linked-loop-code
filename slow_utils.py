import numpy as np

def slow_compute_permissible_parity(Path,cs_decoded_tx_message,J, parityDistribution, toCheck, useWhichMatrix):
    # if path length  = 2
    # then we wanna have parity for section 2. section2Check = 2
    parityDist = parityDistribution[:,toCheck].reshape(1,-1)[0]
    # print("----------parityDist: " + str(parityDist))
    deciders = np.nonzero(parityDist)[0]
    # print("parityDependents = " + str(parityDependents))
    Parity_computed = np.zeros(8, dtype=int)

    focusPath = Path[0]
    for decider in deciders:      # l labels the sections we gonna check to fix toCheck's parities
        gen_mat = matrix_repo(dim=8)[useWhichMatrix[decider][toCheck]] 
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