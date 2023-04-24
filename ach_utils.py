import numpy as np
from scipy.stats import bernoulli


def ach_binary_to_symbol(txBitsParitized, L, K, J):
    tx_symbols = np.zeros((K,L), dtype=int)
    for l in range(L):
        tx_symbols[:,l] = txBitsParitized[:,l*J:(l+1)*J] @ 2**np.arange(J)[::-1].reshape(-1)
    
    # print(f'info_symbols.shape: {tx_symbols.shape}')
    # print(tx_symbols[0:5, :])
    return tx_symbols


def ach_with_error(tx_symbols, L, K, J, p_e):
    for l in range(L):
        applyErrs = np.where(bernoulli.rvs(p_e, size=K))[0]
        Errs = np.random.randint(2**J, size=len(applyErrs))
        tx_symbols[applyErrs,l] = Errs
    
    # rng = np.random.default_rng()
    # tx_symbols = rng.permuted(tx_symbols, axis=0)
    # print(f'After A-Channel, info_symbols.shape: {tx_symbols.shape}')
    # print(tx_symbols[0:5, :])
    return tx_symbols

def ach_with_erasure(tx_symbols, L, K, J, p_e, seed=0):
    np.random.seed(seed=seed)
    tx_temp = np.zeros((K,L*J),dtype=int)
    num_one_outage = 0
    for l in range(L):
        applyErrs = np.where(bernoulli.rvs(p_e, size=K))[0]
        tx_symbols[applyErrs,l] = -1
        tx_temp[:,l] = tx_symbols[:,l]
        tx_symbols_set_l = np.unique(tx_symbols[:,l], axis=-1)
        tx_symbols_set_l = tx_symbols_set_l[tx_symbols_set_l !=-1]
        tx_symbols[0:len(tx_symbols_set_l),l] = tx_symbols_set_l
        tx_symbols[len(tx_symbols_set_l): ,l] = -1
    
    for k in range(K):
        if np.count_nonzero( tx_temp[k,:] == -1) == 1:
            num_one_outage += 1


    # rng = np.random.default_rng()
    # tx_symbols = rng.permuted(tx_symbols, axis=0)
    # print(f'After A-Channel, info_symbols.shape: {tx_symbols.shape}')
    # print(tx_symbols[:, 0])
    return tx_symbols, num_one_outage


def a_plus_ch_with_erasure(tx_symbols, L, K, J, p_e, seed=0):
    np.random.seed(seed=seed)
    tx_symbols_or = tx_symbols.copy()
    tx_temp = np.zeros((K,L*J),dtype=int)
    for l in range(L):
        applyErrs = np.where(bernoulli.rvs(p_e, size=K))[0]
        tx_symbols[applyErrs,l] = -1
        tx_temp[:,l] = tx_symbols[:,l]
        tx_symbols_l = tx_symbols[:,l]
        tx_symbols_l = tx_symbols_l[tx_symbols_l != -1]
        tx_symbols[0:len(tx_symbols_l),l] = tx_symbols_l
        tx_symbols[len(tx_symbols_l): ,l] = -1
    

    one_outage_where = np.zeros((L),dtype=int)
    num_one_outage = 0
    num_no_outage = 0
    for k in range(K):
        num_loss_section = 0
        loss_section = np.zeros((L),dtype=int)
        for l in range(L):
            if tx_temp[:,l].__contains__(tx_symbols_or[k,l]) == False:
                num_loss_section += 1
                loss_section[l] += 1
        
        if num_loss_section == 1:
            num_one_outage += 1
            one_outage_where = one_outage_where + loss_section
        elif num_loss_section == 0:
            num_no_outage += 1

        # if np.count_nonzero( tx_temp[k,:] == -1) == 1:
        #     num_one_outage += 1
        #     one_outage_where[ np.where(tx_temp[k,:] == -1)[0] ] += 1
        # elif np.count_nonzero( tx_temp[k,:] == -1) == 0:
        #     num_no_outage += 1
        
    # rng = np.random.default_rng()
    # tx_symbols = rng.permuted(tx_symbols, axis=0)
    # print(f'After A-Channel, info_symbols.shape: {tx_symbols.shape}')
    # print(tx_symbols[:, 0])

    rng = np.random.default_rng()
    tx_symbols = rng.permuted(tx_symbols, axis=0)

    return tx_symbols, num_one_outage, one_outage_where, num_no_outage