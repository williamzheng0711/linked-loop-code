import numpy as np

"""
A-channel using toy q-ary repetition code. 

Let u be an element of GF(2^16). The repetition encoder 
maps information symbol [u] to [uuu]. 

The decoder recovers [v] if the recovered list contains [vvv].

"""

if __name__ == '__main__':

    # Define simulation parameters
    K = 100
    num_info_bits = 16
    num_coded_bits = 48
    num_coded_sections = 3
    coded_sec_length = 16

    # Generate user messages
    binary_messages = np.random.randint(low=0, high=2, size=(K, num_info_bits))
    print('-'*15+'Generating messages'+'-'*15)
    print(f'binary_messages.shape: {binary_messages.shape}')
    print(binary_messages[0:5, :])

    # Convert 16-bit messages into GF(2^16) symbols
    info_symbols = binary_messages @ 2**np.arange(num_info_bits)[::-1].reshape(-1, 1)
    print('-'*15+'Binary to GF(65536)'+'-'*15)
    print(f'info_symbols.shape: {info_symbols.shape}')
    print(info_symbols[0:5, :])

    # Encode info symbols into codewords
    coded_symbols = np.repeat(info_symbols, repeats=num_coded_sections, axis=1)
    print('-'*15+'Encoding'+'-'*15)
    print(f'coded_symbols.shape: {coded_symbols.shape}')
    print(coded_symbols[0:5, :])

    # A-channel
    rx_coded_symbols = np.unique(coded_symbols, axis=0)
    print('-'*15+'A-Channel'+'-'*15)
    print(f'rx_coded_symbols.shape: {rx_coded_symbols.shape}')
    print(rx_coded_symbols[0:5, :])

    # Jamison Addition
    """
    Note: the A-channel outputs the set of transmitted symbols at each channel
    use. Mathematically, permuting the elements of the set does not change the 
    set. However, permuting the columns of the rx_coded_symbols is a sanity check 
    for me that my recovery algorithm is not exploiting the fact that codewords
    are just rows of the matrix rx_coded_symbols (if all coded symbols are unique). 
    """
    rng = np.random.default_rng()
    rx_coded_symbols = rng.permuted(rx_coded_symbols, axis=0)
    print('-'*15+'Permuting'+'-'*15)
    print(f'rx_coded_symbols.shape: {rx_coded_symbols.shape}')
    print(rx_coded_symbols[0:5, :])

    # Decoding (naive implementation - not optimized at all)
    rx_list = rx_coded_symbols.T.tolist()
    for msg_fragment in rx_list[0].copy():

        # Check if message fragment is contained in all lists
        valid_codeword = True
        for k in range(1, num_coded_sections):
            valid_codeword = valid_codeword and (msg_fragment in rx_list[k])

        # If valid, peel from lists of message fragments
        if valid_codeword:
            for k in range(num_coded_sections):
                rx_list[k].remove(msg_fragment)
    
    # Print number of fragments left per list
    print('-'*15+'Decoding Complete'+'-'*15)
    for i in range(num_coded_sections):
        print(f'len(rx_list[{i}]): {len(rx_list[i])}')

    


