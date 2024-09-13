import numpy as np
from binmatrix import * 
from general_utils import *
from abch_utils import *
from tqdm import * 

L = 16 
J = 16


def randBinMatFullRk(nr, nc):
    flag = 0
    while flag == 0: 
        m = np.random.randint(2, size=(nr, nc))
        binmat = BinMatrix(m)
        if binmat.rank() < min(nr, nc):
            flag = 0
        else: flag = 1
        # flag = 1
    return m


# m = 8,9,...10
def countCollisions(m, K, M, iter, p_e):
    print("**********start*********")
    print(m,p_e,K, iter)
    p_e_set = [p_e]
    # p_e_set = [0.4]
    total_count = np.zeros((len(p_e_set)),dtype=int)
    for index in tqdm(range(iter)): 
        txBits = np.random.randint(low=2, size=(K, m*L)) 
        txCodewords = np.zeros((K,L*J),dtype=int)

        for l in range(L):
            txCodewords[:,l*J:l*J+m] = txBits[:, m*l: m*(l+1)]
            if m < J: 
                who_decides_pl = who_decides_p_sec(L,l,M)
                parity_l = np.zeros((K, J-m), dtype=int)
                for decider in who_decides_pl: 
                    toAdd= ( txBits[:, decider * m : decider * m + m ] @ randBinMatFullRk(nr= m, nc = J-m) ) 
                    # here I need a function that randomly draws a matrix of size m * (J-m) that is full rank
                    parity_l= parity_l+ toAdd
                txCodewords[: ,l*J+m: (l+1)*J]= np.mod(parity_l, 2)
        
        tx_symbols = binary_to_symbol(txCodewords, L, K)

        seed = random.randint(0,100000)
        for p_e in p_e_set: 
            rx_symbols, _, _, _, _, _ = bch_with_erasure(tx_symbols, L, K, p_e, seed=seed)
            count_howmanyB =  np.sum(rx_symbols != -1, axis=0)
            rx_symbols = remove_multiplicity(rx_symbols)
            count_howmanyA = np.sum(rx_symbols != -1, axis=0)
            eachSectionColl = count_howmanyB - count_howmanyA
            total_count[ p_e_set.index(p_e) ] += sum(eachSectionColl)
            # print(sum(eachSectionColl)/(K*L), p_e)

    print("Results are: ")
    for p_e in p_e_set: 
        print(total_count[ p_e_set.index(p_e) ]/iter / (K*L), p_e)

    print("*************end*************")
    return 0

iter = 50000
K = 215
m = 1
countCollisions(m= m, K=K, M=3, iter=iter, p_e=0)
countCollisions(m= m, K=K, M=3, iter=iter, p_e=0.05)
countCollisions(m= m, K=K, M=3, iter=iter, p_e=0.1)
countCollisions(m= m, K=K, M=3, iter=iter, p_e=0.15)
countCollisions(m= m, K=K, M=3, iter=iter, p_e=0.2)
countCollisions(m= m, K=K, M=3, iter=iter, p_e=0.25)
countCollisions(m= m, K=K, M=3, iter=iter, p_e=0.3)
countCollisions(m= m, K=K, M=3, iter=iter, p_e=0.35)
countCollisions(m= m, K=K, M=3, iter=iter, p_e=0.4)