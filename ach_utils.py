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

def ach_with_deletion(tx_symbols, L, K, J, p_e, seed=0):
    np.random.seed(seed=seed)
    for l in range(L):
        applyErrs = np.where(bernoulli.rvs(p_e, size=K))[0]
        tx_symbols[applyErrs,l] = -1
    
    # rng = np.random.default_rng()
    # tx_symbols = rng.permuted(tx_symbols, axis=0)
    # print(f'After A-Channel, info_symbols.shape: {tx_symbols.shape}')
    # print(tx_symbols[0:5, :])
    return tx_symbols