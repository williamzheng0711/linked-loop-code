from numpy import pi, sqrt
import numpy as np
from scipy.special import erf
from scipy.stats import norm as normal
from scipy.stats import rice, rayleigh
from scipy.integrate import quad_vec
from pyfht import block_sub_fht

def matrix_repo(dim): 
    if dim == 2:
        return [ [[1,0],[0,1]], 
                 [[1,1],[0,1]], 
                 [[0,1],[1,0]], 
                 [[1,0],[1,1]], 
                 [[1,1],[1,0]], 
                 [[0,1],[1,1]] ]
    elif dim == 3: 
        return [ [[1, 0, 0],
                  [0, 1, 0],
                  [0, 0, 1]],
            
                 [[0, 1, 0],
                  [1, 1, 0],
                  [0, 1, 1]],

                 [[0, 1, 1],
                  [0, 0, 1],
                  [1, 0, 0]],

                 [[0, 0, 1],
                  [1, 0, 0],
                  [1, 1, 0]],

                 [[0, 0, 1],
                  [0, 1, 1],
                  [1, 0, 1]] ]
    elif dim == 4:
        return [ [[1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]],

                 [[0, 1, 0, 1],
                  [0, 1, 1, 0],
                  [0, 1, 0, 0],
                  [1, 1, 1, 0]],

                 [[1, 0, 1, 1],
                  [0, 1, 1, 0],
                  [1, 0, 0, 0],
                  [1, 0, 0, 1]], 

                 [[0, 0, 0, 1],
                  [1, 0, 1, 0],
                  [0, 1, 1, 1],
                  [1, 0, 0, 1]],  

                 [[0, 1, 0, 0],
                  [0, 1, 0, 1],
                  [0, 1, 1, 0],
                  [1, 0, 1, 1]] ]
    elif dim == 7: 
        return [ [[1, 0, 0, 0, 0, 0, 0],
                  [0, 1, 0, 0, 0, 0, 0],
                  [0, 0, 1, 0, 0, 0, 0],
                  [0, 0, 0, 1, 0, 0, 0],
                  [0, 0, 0, 0, 1, 0, 0],
                  [0, 0, 0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 0, 0, 1]],

                 [[0, 0, 1, 0, 1, 0, 0],
                  [0, 1, 0, 0, 1, 0, 1],
                  [1, 1, 0, 1, 1, 0, 0],
                  [1, 0, 1, 1, 0, 1, 0],
                  [1, 0, 0, 0, 1, 0, 1],
                  [0, 1, 0, 1, 0, 1, 0],
                  [1, 0, 1, 1, 1, 1, 1]],

                 [[0, 0, 0, 0, 0, 0, 1],
                  [1, 0, 1, 1, 0, 1, 1],
                  [1, 1, 0, 1, 1, 0, 0],
                  [0, 0, 1, 0, 0, 0, 0],
                  [1, 0, 1, 0, 1, 1, 1],
                  [0, 1, 0, 1, 1, 0, 1],
                  [0, 1, 1, 0, 0, 1, 1]] ]
    elif dim == 8:
        return [   [[0, 1, 1, 0, 0, 1, 0, 1],
                    [0, 0, 0, 0, 1, 1, 0, 1],
                    [0, 1, 1, 1, 0, 1, 0, 1],
                    [1, 0, 1, 0, 0, 1, 1, 1],
                    [0, 0, 1, 0, 0, 1, 1, 1],
                    [0, 1, 1, 1, 0, 1, 0, 0],
                    [0, 0, 1, 0, 1, 0, 0, 0],
                    [0, 0, 1, 0, 1, 1, 1, 1]],

                    [[1, 1, 0, 0, 1, 0, 1, 1],
                    [1, 0, 0, 1, 1, 1, 0, 0],
                    [1, 1, 0, 1, 0, 0, 0, 1],
                    [0, 0, 1, 0, 1, 0, 1, 0],
                    [0, 1, 0, 1, 1, 0, 1, 0],
                    [1, 0, 1, 0, 0, 1, 0, 1],
                    [1, 1, 0, 0, 1, 0, 1, 0],
                    [0, 1, 0, 0, 1, 1, 1, 0]],

                    [[1, 0, 1, 1, 0, 0, 1, 1],
                    [0, 1, 1, 1, 0, 0, 0, 0],
                    [1, 0, 0, 0, 1, 1, 1, 0],
                    [1, 1, 0, 1, 0, 1, 1, 0],
                    [0, 0, 1, 1, 0, 0, 0, 1],
                    [1, 0, 1, 0, 1, 1, 1, 0],
                    [0, 1, 0, 1, 0, 0, 1, 1],
                    [0, 0, 1, 0, 0, 1, 0, 1]],

                    [[0, 0, 1, 1, 1, 0, 0, 1],
                    [0, 1, 0, 1, 0, 0, 0, 0],
                    [1, 1, 0, 0, 1, 1, 1, 0],
                    [1, 0, 1, 1, 0, 1, 0, 0],
                    [0, 1, 0, 1, 0, 0, 1, 1],
                    [0, 1, 0, 0, 1, 1, 1, 0],
                    [1, 0, 1, 1, 0, 1, 0, 1],
                    [1, 1, 0, 0, 0, 1, 0, 1]],

                    [[1, 0, 1, 1, 0, 0, 0, 0],
                    [1, 0, 1, 1, 0, 0, 1, 1],
                    [1, 0, 0, 0, 1, 1, 1, 0],
                    [1, 1, 0, 1, 0, 0, 1, 1],
                    [0, 0, 1, 1, 0, 0, 1, 0],
                    [0, 1, 0, 0, 1, 1, 0, 1],
                    [0, 1, 1, 0, 1, 1, 0, 1],
                    [1, 0, 0, 0, 0, 1, 0, 1]],

                    [[1, 0, 0, 1, 0, 1, 0, 1],
                    [0, 1, 1, 1, 0, 0, 1, 1],
                    [1, 0, 1, 0, 1, 1, 0, 0],
                    [0, 1, 0, 0, 1, 0, 0, 1],
                    [1, 1, 0, 1, 1, 0, 1, 0],
                    [1, 1, 0, 1, 1, 0, 1, 1],
                    [0, 1, 0, 0, 0, 0, 0, 1],
                    [0, 1, 0, 1, 0, 1, 0, 0]],

                    [[0, 1, 0, 1, 1, 0, 0, 0],
                    [1, 1, 0, 0, 0, 1, 1, 1],
                    [0, 0, 1, 0, 1, 0, 0, 1],
                    [1, 0, 1, 1, 1, 0, 1, 0],
                    [0, 1, 0, 0, 0, 1, 1, 1],
                    [1, 0, 0, 1, 0, 0, 1, 0],
                    [1, 0, 1, 1, 1, 0, 0, 0],
                    [1, 0, 0, 1, 1, 0, 1, 1]],

                                                ]

def matrix_inv_repo(dim):
    if dim == 8:
        return [   [[0, 0, 0, 1, 1, 0, 0, 0],
                    [1, 1, 0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 1, 0, 1, 1],
                    [1, 0, 1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 0, 0, 1],
                    [0, 1, 1, 0, 1, 1, 0, 1],
                    [0, 1, 0, 0, 1, 0, 1, 0],
                    [0, 0, 1, 0, 0, 1, 0, 0]],

                   [[0, 0, 1, 1, 1, 1, 1, 1],
                    [1, 0, 1, 0, 1, 0, 0, 0],
                    [1, 0, 0, 0, 0, 1, 0, 1],
                    [0, 0, 1, 1, 0, 1, 0, 1],
                    [0, 1, 1, 1, 0, 1, 1, 0],
                    [0, 0, 1, 1, 1, 1, 0, 0],
                    [1, 1, 1, 0, 0, 0, 1, 1],
                    [1, 0, 0, 0, 0, 0, 1, 0]],

                   [[0, 0, 1, 1, 0, 1, 1, 1],
                    [1, 0, 0, 1, 0, 0, 0, 1],
                    [0, 0, 1, 0, 0, 1, 0, 0],
                    [1, 1, 1, 1, 0, 1, 0, 1],
                    [0, 1, 0, 1, 0, 1, 0, 0],
                    [1, 1, 1, 1, 1, 1, 0, 0],
                    [1, 0, 1, 1, 1, 1, 1, 1],
                    [1, 1, 0, 1, 1, 0, 0, 1]],

                   [[0, 0, 1, 0, 0, 1, 0, 0],
                    [1, 1, 1, 1, 1, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 1, 1],
                    [1, 0, 1, 1, 1, 0, 0, 0],
                    [0, 1, 1, 0, 1, 0, 0, 1],
                    [1, 1, 0, 0, 1, 1, 1, 1],
                    [0, 1, 0, 1, 1, 0, 1, 0],
                    [0, 0, 0, 1, 0, 0, 1, 0]],

                   [[1, 0, 1, 1, 0, 0, 1, 0],
                    [0, 1, 0, 1, 0, 1, 1, 0],
                    [0, 0, 0, 0, 0, 1, 1, 0],
                    [0, 0, 1, 1, 0, 1, 0, 0],
                    [1, 1, 1, 0, 0, 0, 0, 1],
                    [0, 1, 0, 0, 1, 0, 0, 1],
                    [0, 0, 1, 1, 1, 0, 1, 0],
                    [1, 1, 1, 1, 1, 0, 1, 0]],

                   [[1, 0, 0, 0, 0, 0, 1, 1],
                    [0, 0, 0, 0, 1, 1, 1, 0],
                    [1, 1, 0, 1, 0, 1, 0, 1],
                    [0, 1, 1, 0, 1, 0, 1, 1],
                    [0, 0, 0, 1, 0, 0, 1, 0],
                    [0, 1, 1, 0, 0, 1, 0, 0],
                    [1, 1, 1, 1, 1, 1, 0, 0],
                    [0, 0, 0, 0, 1, 1, 0, 0]], 

                   [[0, 1, 0, 0, 1, 0, 0, 0],
                    [1, 1, 1, 0, 1, 1, 1, 1],
                    [0, 0, 1, 0, 0, 1, 0, 1],
                    [0, 1, 0, 1, 1, 1, 1, 0],
                    [0, 0, 1, 1, 0, 0, 0, 1],
                    [1, 1, 0, 0, 0, 0, 0, 1],
                    [0, 0, 0, 1, 0, 0, 1, 0],
                    [0, 0, 1, 1, 0, 1, 0, 0]] ]

def get_parity_involvement_matrix(L, windowSize, messageLen):
    """
    Construct the parity involvement matrix.

    Parameters
    ----------
    L (int): number of sections in codeword

    Returns
    -------
    parityInvolved : ndarray (L by L matrix)
        For each row i, the j-th entry equals w/L(=8) if w(i) is involved in the construction of p(j). 
        Otherwise equals 0.

    Notes
    -----
    E.g., 
        parityInvolved[0] = [0,8,8,8,8,0,0,0,0,0,0,0,0,0,0,0]. As w(0) is involved in determining p(1), p(2), p(3) and p(4).
        
        For the same reason, if messageLen=8 windowSize=4, parityInvolved[1] = [0,0,8,8,8,8,0,0,0,0,0,0,0,0,0,0], etc..

    """
    parityInvolved = np.zeros((L,L), dtype=int)
    offsets = np.arange(1, windowSize+1)
    for l in range(L):
        parityInvolved[l, (l + offsets) % L] = messageLen
    return parityInvolved

def get_G_matrices(parityInvolved):
    """
    Construct a index matrix that specifies G_{i,j} matrices for all valid (i,j) pair.

    Parameters
    ----------
    parityInvolved : ndarray (L by L matrix)
        For each row i, the j-th entry equals w/L(=8) if w(i) involves the construction of p(j). 
        Otherwise equals 0.

    Returns
    -------
    whichGMatrix : ndarray (L by L matrix)
        Only (i,j) s.t. parityInvolved[i][j] != 0 matters. Otherwise whichGMatrix[i][j] = -1.
        
        For (i,j) of our interest, whichGMatrix[i][j] returns a code (an index) for some specific G_{i,j} matrix stored in matrix_repo(8).
        
        Where G_{i,j} matrix is the parity generating matrix needed to 
        calculate the contribution of w(i) while calculating p(j)

    Notes
    -----
    For instance, 
        whichGMatrix[0] can be [-1  3  1  0  0 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1]. 
        Only those >-1 entries matter.
    """

    # Extract L from parityInvolved
    L = parityInvolved.shape[0]

    # Obtain dim as well as window size
    dim = np.max(parityInvolved[0, :]).astype(int)
    windowSize = np.sum(parityInvolved[0, :]) // dim 

    # Obtain num choices
    numChoices = len(matrix_repo(dim))

    # Construct whichGMatrix
    whichGMatrix = -1*np.ones((L, L), dtype=int)
    whichGMatrix[np.nonzero(parityInvolved)] = np.random.randint(low=0, high=numChoices, size=(L*windowSize, ))
    return whichGMatrix

def convert_bits_to_sparse_Rayleigh(encoded_tx_message, L, J, M, K, sigma_R):
    encoded_tx_message_sparse=np.zeros((L*M, 1),dtype=float)
    # Here we generate K iid random variables, each ~ Rayleigh(sigma_R)
    fading_coefficients = np.random.rayleigh(scale=sigma_R, size=K).reshape(-1, 1) 
    for i in range(L):
        idx_nonzero = encoded_tx_message[:,i*J:(i+1)*J] @ 2**np.arange(J)[::-1].reshape(-1, 1)
        encoded_tx_message_sparse[i*M + idx_nonzero, 0] = fading_coefficients

    return encoded_tx_message_sparse

# This serves as an (common) inner code
def sparc_codebook(L, M, n):
    Ax, Ay, _ = block_sub_fht(n, M, L, seed=0, ordering=None)
    def Ab(b):
        return Ax(b).reshape(-1, 1)/ np.sqrt(n)
    def Az(z):
        return Ay(z.flatten()).reshape(-1, 1)/ np.sqrt(n)
    return Ab, Az

def amp_prior_art_Rayleigh(y, sigma_n, P, L, M, T, Ab, Az, p0, K, sigma_R, convertToBeta):
    n = y.size
    beta = np.zeros((L*M, 1))
    z = y.copy()
    Phat = n*P/L
    dl = np.sqrt(Phat)    
    for t in range(T):
        
        tau = np.sqrt(np.sum(z**2)/n)
        print(" | now is iter" + str(t) + " and tau is: " + str(tau))

        # effective observation
        r = (np.sqrt(Phat)*beta + Az(z)).astype(np.longdouble) 
        # print("r[0,1]=" +  str(r[0:2]))
        
        # denoiser
        a = (tau**2+dl**2*sigma_R**2)/(2*tau**2*sigma_R**2)
        b = dl / tau**2 * r

        # Numerator
        Nume =       1/(sigma_R**2*np.sqrt(2*pi)) * np.exp(-r**2/(2*tau**2)) * ( b/(4*a**2)  + np.exp(b**2/(4*a)) * (2*a + b**2)/(8*a**2) * np.sqrt(pi/a) * (1-erf(-b/(2*np.sqrt(a)), dtype=float)) )  
        # print("手算的： " + str(Nume[0:2]))
        # Nume_int_part, _ = quad_vec(f = lambda h:   h*normal.pdf((r-dl*h)/tau) * rayleigh.pdf(h, 0, sigma_R), a=0, b=np.Infinity) 
        # print("機器算的： " + str(Nume_int_part[0:2]))


        # Deno is TWO parts: 
        Deno_hard =  1/(sigma_R**2*np.sqrt(2*pi)) * np.exp(-r**2/(2*tau**2)) * ( 1/(2*a) + np.exp(b**2/(4*a)) * b/(4*a) * np.sqrt(pi/a) * (1-erf(-b/(2*np.sqrt(a)), dtype=float)))
        Deno_easy =  normal.pdf(r/tau)
        # print("Deno手算的： " + str(Deno_hard[0:2]))
        # Deno_hard_int,_ = quad_vec(f=lambda h: normal.pdf((r-dl*h)/tau) * rayleigh.pdf(h, 0, sigma_R), a=0, b=np.Infinity) 
        # print("Deno機器算的： " + str(Deno_hard_int[0:2]))

        beta =  p0 * Nume  / ( (1-p0) * Deno_easy + p0 * Deno_hard ) 

        # residual
        # E_h2s2 = E[h^2 s^2 | r], in desmos were denoted as a_this and b_this
        Nume_square_term = 1/(sigma_R**2 * np.sqrt(2*pi)) * np.exp(-r**2/(2*tau**2)) * ( (b**2/(8*a**3) + 1/(2*a**2)) + np.exp(b**2/(4*a))*(3*b/(8*a**2) + b**3/(16*a**3)) * np.sqrt(pi/a) * (1-erf(-b/(2*np.sqrt(a)), dtype=float)) ) 
        # print("square_term 手算的： " + str(Nume_square_term[0:2]))
        # Nume_square_int, _ = quad_vec(f=lambda h: h**2 * normal.pdf((r-dl*h)/tau) * rayleigh.pdf(h, 0, sigma_R), a=0, b=np.Infinity)      
        # print("square term 電腦算的: " + str(Nume_square_int[0:2]))

        E_h2s2 = p0 * Nume_square_term / ( (1-p0) * Deno_easy + p0 * Deno_hard ) 
        z = y - np.sqrt(Phat)*Ab(beta) + (z/n) * (Phat/tau**2) * sum(  E_h2s2 - beta**2  )
    
    if (convertToBeta):
        # In the last round
        coeff_erf = -(sigma_R*dl)/(tau*np.sqrt(2*(sigma_R**2*dl**2+tau**2)))
        beta_Nume_raw = 1/(sigma_R**2*np.sqrt(2*pi)) * np.exp(-(r**2)/(2*tau**2))*(sigma_R**2*tau**2)/(sigma_R**2*dl**2+tau**2) + 1/(2*np.sqrt(pi))*np.exp(-r**2/(2*(dl**2*sigma_R**2+tau**2))) * (sigma_R*dl*r*tau)/(dl**2*sigma_R**2+tau**2)**(3/2) * (np.sqrt(pi)*( 1-erf(coeff_erf*r, dtype=float)) ) 
        # print("by hand： " + str(beta_Nume_raw[0:2]))
        # Nume_raw_int, _ = quad_vec(f = lambda h:   normal.pdf((r-dl*h)/tau) * rayleigh.pdf(h, 0, sigma_R), a=0, b=np.Infinity) 
        # print("機器算的： " + str(Nume_raw_int[0:2]))

        beta = (p0 * beta_Nume_raw) / (p0 * beta_Nume_raw + (1-p0)*normal.pdf(r/tau) )

    return beta

def check_if_identical_msgs(Paths, cs_decoded_tx_message, L,J,messageLen):   
    msg_bits = extract_msg_bits(Paths,cs_decoded_tx_message, L,J,messageLen)
    flag = (msg_bits == msg_bits[0]).all()    
    return flag

def extract_msg_bits(Paths,cs_decoded_tx_message, L,J,messageLen):
    msg_bits = np.empty(shape=(0,0), dtype=int)
    L1 = Paths.shape[0]
    for i in range(L1):
        msg_bit=np.empty(shape=(0,0), dtype=int)
        path = Paths[i].reshape(1,-1)
        for j in range(path.shape[1]):
            msg_bit = np.hstack((msg_bit,cs_decoded_tx_message[path[0,j],J*j:J*j+messageLen].reshape(1,-1))) if msg_bit.size else cs_decoded_tx_message[path[0,j],J*(j):J*(j)+messageLen]
            msg_bit=msg_bit.reshape(1,-1)
        msg_bits = np.vstack((msg_bits,msg_bit)) if msg_bits.size else msg_bit           
    return msg_bits

def analyze_genie_metrics(decTempBETA, L, J, listSize, txBitsParitized, K):
    """
    Compute genie metrics

        Parameters:
            decTempBETA (ndarray): current estimate of sparse coded vector
            L (int): number of sections in the codeword
            J (int): number of bits per codeword section
            listSize (int): how many entries to retain per section of the codeword
            txBitsParitized (ndarray): outer encoded tx messages
            K (int): true number of active users

        Returns:
            <none>
    """
    thisTimeGenie = 0
    decOutMsg = convert_sparse_to_bits(decTempBETA, L, J, listSize, ) 
    error_box = []
    for i in range(K):
        oneOutageSection = 0
        flag_i = 1
        num_not_match_i = 0
        for l in range(L):                
            tmp = np.equal(txBitsParitized[i, l*J: (l+1)*J ], 
                            decOutMsg[:, l*J: (l+1)*J ]).all(axis=1).any()
            if not tmp:
                flag_i = 0
                num_not_match_i += 1
                if num_not_match_i == 1: 
                    oneOutageSection = l
        thisTimeGenie += flag_i
        if not flag_i:
            error_box.append(num_not_match_i)
            if (num_not_match_i == 1): 
                print(" | Some one-outage message has that one-outage at section:"  + str(oneOutageSection))
    print(" | genie recovers " + str(thisTimeGenie) +" out of " + str(K))
    print(" | How many sections do they lose? " + str(error_box))

def get_sig_values_and_positions(decTempBETA, L, J, listSize):
    M = 2**J
    decBetaSignificants = np.zeros((L, listSize))
    decBetaSignificantsPos = np.zeros((L, listSize), dtype=int)
    for l in range(L):
        section_l = decTempBETA[l*M:(l+1)*M]
        idx_top_vals = np.argpartition(section_l.flatten(), -listSize)[-listSize:]
        decBetaSignificants[l, :] = section_l[idx_top_vals].flatten()
        decBetaSignificantsPos[l, :] = idx_top_vals
    return decBetaSignificants.T, decBetaSignificantsPos.T

def convert_sparse_to_bits(cs_decoded_tx_message_sparse, L, J, listSize):
    M = 2**J
    cs_decoded_tx_message = np.zeros((listSize, L*J), dtype=int)
    for i in range(L):
        A = cs_decoded_tx_message_sparse[i*M:(i+1)*M]
        B = np.argpartition(A.flatten(), -listSize)[-listSize:]
        cs_decoded_tx_message[:,i*J:(i+1)*J]=np.array([list(np.binary_repr(int(x),J)) for x in B], dtype=int)    
    return cs_decoded_tx_message


def check_phase_1(txBits, rxBits, name):
    # Check how many are correct amongst the recover (recover means first phase). No need to change.
    thisIter = 0
    txBits_remained_llc = np.empty(shape=(0,0))
    for i in range(txBits.shape[0]):
        incre = 0
        incre = np.equal(txBits[i,:],rxBits).all(axis=1).any()
        thisIter += int(incre)
        if (incre == False):
            txBits_remained_llc = np.vstack( (txBits_remained_llc, txBits[i,:]) ) if txBits_remained_llc.size else  txBits[i,:]
    print(" | In phase 1, " + str(name) + " decodes " + str(thisIter) + " true message out of " +str(rxBits.shape[0]))
    # print(" - " + str(name) + " Phase 1 is done.")
    return txBits_remained_llc