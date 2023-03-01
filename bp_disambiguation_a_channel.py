import numpy as np
import factor_graph_generation as FGG

"""
CCS-AMP style BP disambiguation over A-channel.
"""

if __name__ == '__main__':

    # Simulation Parameters
    K = 100
    num_sections = 16
    sec_length = 2**16
    num_info_bits = 128

    # Initialize outer code
    outer_code = FGG.Triadic8(16)

    # Generate user messages
    user_binary_messages = np.random.randint(2, size=(K, num_info_bits))
    print(f'user_binary_messages.shape: {user_binary_messages.shape}')

    # Encode user messages
    user_codewords = outer_code.encodemessages(user_binary_messages)
    for cdwd in user_codewords:
        assert not np.isscalar(outer_code.testvalid(cdwd))
    print(f'user_codewords.shape: {user_codewords.shape}')

    # A-Channel
    a_chnl_output = np.sum(user_codewords, axis=0)
    a_chnl_output = np.minimum(a_chnl_output, 1)
    print(f'a_chnl_output.shape: {a_chnl_output.shape}')

    # extract valid codewords
    rx_user_codewords = outer_code.decoder(a_chnl_output, K)
    rx_user_codewords = np.array(rx_user_codewords)
    print(f'rx_user_codewords.shape: {rx_user_codewords.shape}')

    # Count number of matches
    num_matches = FGG.numbermatches(user_codewords, rx_user_codewords, K)
    print(f'Recovered {num_matches}/{K} codewords. ')
