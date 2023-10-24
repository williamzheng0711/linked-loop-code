import numpy as np
import factor_graph_generation as FGG

def LDPC_encode_to_symbol(txBits, L, K, J, outer_code):
    user_codewords = outer_code.encodemessages(txBits)
    for cdwd in user_codewords:
        assert not np.isscalar(outer_code.testvalid(cdwd))

    encoded_tx_symbols = np.zeros((K, L), dtype=int)    # (100,256)
    for k in range(K):
        for l in range(L):
            temp = np.array(user_codewords[k, l*2**J:(l+1)*2**J], dtype=int)
            encoded_tx_symbols[k,l] = 2**J - np.argmax(temp)
    
    # print(encoded_tx_symbols)
    # print(f'encoded_tx_symbols.shape: {encoded_tx_symbols.shape}')
    return encoded_tx_symbols, user_codewords

def LDPC_symbols_to_bits(L, J, rx_coded_symbols_ldpc, K, channel):
    unioned_cdwds_ldpc = np.zeros(L*2**J, dtype=int)
    for l in range(L):
        temp_l = np.zeros(2**J, dtype=int)
        for k in range(K):
            if rx_coded_symbols_ldpc[k,l] >= 0:
                temp_l[2**J - rx_coded_symbols_ldpc[k,l]] += 1
        if channel == "A":
            temp_l = np.minimum(temp_l , 1)
            unioned_cdwds_ldpc[l*2**J : (l+1)*2**J] = temp_l
        elif channel == "B":
            unioned_cdwds_ldpc[l*2**J : (l+1)*2**J] = temp_l
    # print(np.sum(unioned_cdwds_ldpc))         # should be about 1600
    return unioned_cdwds_ldpc