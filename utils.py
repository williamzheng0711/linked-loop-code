from numpy import pi, sqrt
import numpy as np
from scipy.special import erf
from scipy.stats import norm as normal
from scipy.stats import rice, rayleigh
from scipy.integrate import quad_vec
import timeit


def matrix_repo(dim): 
    if dim == 2:
        return [ [[1,0],[0,1]], 
                 [[1,1],[0,1]], 
                 [[0,1],[1,0]], 
                 [[1,0],[1,1]], 
                 [[1,1],[1,0]], 
                 [[0,1],[1,1]] ]
    if dim == 3: 
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

    if dim == 4:
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

    if dim == 7: 
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

    if dim == 8:
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

def get_parity_involvement_matrix(L):
    """
    Construct the parity involvement matrix.

    Parameters
    ----------
    None.

    Returns
    -------
    parityInvolved : ndarray (L by L matrix)
        For each row i, the j-th entry equals w/L(=8) if w(i) involves the construction of p(j). 
        Otherwise equals 0.

    Notes
    -----
    E.g., 
        parityInvolved[0] = [0,8,8,8,8,0,0,0,0,0,0,0,0,0,0,0]. As w(0) is involved in determining p(1), p(2), p(3) and p(4).
        
        For the same reason, parityInvolved[1] = [0,0,8,8,8,8,0,0,0,0,0,0,0,0,0,0], etc..

    """
    parityInvolved = np.zeros((L,L),dtype=int)
    for l in np.arange(L):
        for i in [1,2,3,4]:
            parityInvolved[l][ (l+i) % L] = 8     

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

    whichGMatrix = -1 * np.ones((16,16),dtype=int)
    for row in np.arange(0,16):
        for col in np.arange(0, 16):
            if parityInvolved[row][col]!=0:
                dim = parityInvolved[row][col]
                choices = matrix_repo(dim)
                whichGMatrix[row][col] = np.random.randint(low=0, high=len(choices))
    return whichGMatrix



def convert_bits_to_sparse_Rayleigh(encoded_tx_message,L,J,K, sigma_R):
    encoded_tx_message_sparse=np.zeros((L*2**J,1),dtype=float)
    # Here we generate K iid random variables, each ~ Rayleigh(sigma_R)
    fading_coefficients = np.random.rayleigh(scale = sigma_R, size=K) 
    for i in range(L):
        A = encoded_tx_message[:,i*J:(i+1)*J]
        B = A.dot(2**np.arange(J)[::-1]).reshape([K,1])
        for k in range(K):
            encoded_tx_message_sparse[i*2**J+B[k]] += fading_coefficients[k]      
    return encoded_tx_message_sparse


# This serves as an (common) inner code
def sparc_codebook(L, M, n):
    Ax, Ay, _ = block_sub_fht(n, M, L, ordering=None)
    def Ab(b):
        return Ax(b).reshape(-1, 1)/ np.sqrt(n)
    def Az(z):
        return Ay(z).reshape(-1, 1)/ np.sqrt(n)
    return Ab, Az



def block_sub_fht(n, m, l, seed=0, ordering=None, new_embedding=False):
    """
    As `sub_fht`, but computes in `l` blocks of size `n` by `m`, potentially
    offering substantial speed improvements.

    n: number of rows
    m: number of columns per block
    l: number of blocks

    It is most efficient (though not required) when max(m,n+1) is a power of 2.

    seed: determines choice of random matrix
    ordering: optional (l, n) shaped array of row indices in [1, max(m, n)] to
              implement subsampling; generated by seed if not specified, but
              may be given to speed up subsequent runs on the same matrix.

    Returns (Ax, Ay, ordering):
        Ax(x): computes A.x (of length n), with x having length l*m
        Ay(y): computes A'.y (of length l*m), with y having length n
        ordering: the ordering in use, which may have been generated from seed
    """
    assert n > 0, "n must be positive"
    assert m > 0, "m must be positive"
    assert l > 0, "l must be positive"

    if ordering is not None:
        assert ordering.shape == (l, n)
    else:
        if new_embedding:
            w = 2**int(np.ceil(np.log2(max(m+1, n+1))))
        else:
            w = 2**int(np.ceil(np.log2(max(m, n+1))))
            # w = 2^16 = 一個section的長度 = 2^J
        rng = np.random.RandomState(seed)
        ordering = np.empty((l, n), dtype=np.uint32)
        idxs = np.arange(1, w, dtype=np.uint32)
        for ll in range(l):
            rng.shuffle(idxs)
            ordering[ll] = idxs[:n]
        
        # 跑完之後 ordering 是一個 (16,30000) 的矩陣 Aka 每個section有30000個index 這些index 採樣自[1,65536]


    def Ax(x):
        # print(m)
        # print(x.size)
        # print("l="+str(l))
        assert x.size == l*m
        out = np.zeros(n)
        for ll in range(l):
            ax, ay, _ = sub_fht(n, m, ordering=ordering[ll],
                                new_embedding=new_embedding)
            out += ax(x[ll*m:(ll+1)*m])
        return out

    def Ay(y):
        assert y.size == n
        out = np.empty(l*m)
        for ll in range(l):
            ax, ay, _ = sub_fht(n, m, ordering=ordering[ll],
                                new_embedding=new_embedding)
            out[ll*m:(ll+1)*m] = ay(y)
        return out

    return Ax, Ay, ordering

def fht(u):
    """
    Perform fast Hadamard transform of u, in-place.
    Note len(u) must be a power of two.
    """
    N = len(u)
    i = N>>1
    while i:
        for j in range(N):
            if (i&j) == 0:
                temp = u[j]
                u[j] += u[i|j]
                u[i|j] = temp - u[i|j]
        i>>= 1


def sub_fht(n, m, seed=0, ordering=None, new_embedding=False):
    """
    Returns functions to compute the sub-sampled Walsh-Hadamard transform,
    i.e., operating with a wide rectangular matrix of random +/-1 entries.

    n: number of rows
    m: number of columns

    It is most efficient (but not required) for max(m,n+1) to be a power of 2.

    seed: determines choice of random matrix
    ordering: optional n-long array of row indices in [1, max(m,n)] to
              implement subsampling; generated by seed if not specified,
              but may be given to speed up subsequent runs on the same matrix.

    Returns (Ax, Ay, ordering):
        Ax(x): computes A.x (of length n), with x having length m
        Ay(y): computes A'.y (of length m), with y having length n
        ordering: the ordering in use, which may have been generated from seed
    """
    assert n > 0, "n must be positive"
    assert m > 0, "m must be positive"
    if new_embedding:
        w = 2**int(np.ceil(np.log2(max(m+1, n+1))))
    else:
        w = 2**int(np.ceil(np.log2(max(m, n+1))))
        # still w = 65536

    if ordering is not None:
        assert ordering.shape == (n,)
    else:
        rng = np.random.RandomState(seed)
        # seed = 0 hence rng is same as before is RandomState(MT19937)
        idxs = np.arange(1, w, dtype=np.uint32)
        rng.shuffle(idxs)
        ordering = idxs[:n]

    def Ax(x):
        assert x.size == m, "x must be m long"
        y = np.zeros(w)
        # y is w = 65536 long vector
        if new_embedding:
            y[w-m:] = x.reshape(m)
        else:
            y[:m] = x.reshape(m)
            # m = 65536
        fht(y)
        return y[ordering]

    def Ay(y):
        assert y.size == n, "input must be n long"
        x = np.zeros(w)
        x[ordering] = y.reshape(n)
        fht(x)
        if new_embedding:
            return x[w-m:]
        else:
            return x[:m]

    return Ax, Ay, ordering



def amp_prior_art_Rayleigh(y, σ_n, P, L, M, T, Ab, Az, p0, K, sigma_R, convertToBeta):
    n = y.size
    β = np.zeros((L*M, 1))
    z = y
    Phat = n*P/L
    dl = np.sqrt(Phat)    
    for t in range(T):
        
        τ = np.sqrt(np.sum(z**2)/n)
        print(" | now is iter" + str(t) + " and tau is: " + str(τ))

        # effective observation
        r = (np.sqrt(Phat)*β + Az(z)).astype(np.longdouble) 
        # print("r[0,1]=" +  str(r[0:2]))
        
        # denoiser
        a = (τ**2+dl**2*sigma_R**2)/(2*τ**2*sigma_R**2)
        b = dl / τ**2 * r

        # Numerator
        Nume =       1/(sigma_R**2*np.sqrt(2*pi)) * np.exp(-r**2/(2*τ**2)) * ( b/(4*a**2)  + np.exp(b**2/(4*a)) * (2*a + b**2)/(8*a**2) * np.sqrt(pi/a) * (1-erf(-b/(2*np.sqrt(a)), dtype=float)) )  
        # print("手算的： " + str(Nume[0:2]))
        # Nume_int_part, _ = quad_vec(f = lambda h:   h*normal.pdf((r-dl*h)/τ) * rayleigh.pdf(h, 0, sigma_R), a=0, b=np.Infinity) 
        # print("機器算的： " + str(Nume_int_part[0:2]))


        # Deno is TWO parts: 
        Deno_hard =  1/(sigma_R**2*np.sqrt(2*pi)) * np.exp(-r**2/(2*τ**2)) * ( 1/(2*a) + np.exp(b**2/(4*a)) * b/(4*a) * np.sqrt(pi/a) * (1-erf(-b/(2*np.sqrt(a)), dtype=float)))
        Deno_easy =  normal.pdf(r/τ)
        # print("Deno手算的： " + str(Deno_hard[0:2]))
        # Deno_hard_int,_ = quad_vec(f=lambda h: normal.pdf((r-dl*h)/τ) * rayleigh.pdf(h, 0, sigma_R), a=0, b=np.Infinity) 
        # print("Deno機器算的： " + str(Deno_hard_int[0:2]))

        β =  p0 * Nume  / ( (1-p0) * Deno_easy + p0 * Deno_hard ) 

        # residual
        # E_h2s2 = E[h^2 s^2 | r], in desmos were denoted as a_this and b_this
        Nume_square_term = 1/(sigma_R**2 * np.sqrt(2*pi)) * np.exp(-r**2/(2*τ**2)) * ( (b**2/(8*a**3) + 1/(2*a**2)) + np.exp(b**2/(4*a))*(3*b/(8*a**2) + b**3/(16*a**3)) * np.sqrt(pi/a) * (1-erf(-b/(2*np.sqrt(a)), dtype=float)) ) 
        # print("square_term 手算的： " + str(Nume_square_term[0:2]))
        # Nume_square_int, _ = quad_vec(f=lambda h: h**2 * normal.pdf((r-dl*h)/τ) * rayleigh.pdf(h, 0, sigma_R), a=0, b=np.Infinity)      
        # print("square term 電腦算的: " + str(Nume_square_int[0:2]))

        E_h2s2 = p0 * Nume_square_term / ( (1-p0) * Deno_easy + p0 * Deno_hard ) 
        z = y - np.sqrt(Phat)*Ab(β) + (z/n) * (Phat/τ**2) * sum(  E_h2s2 - β**2  )
    
    if (convertToBeta):
        # In the last round
        coeff_erf = -(sigma_R*dl)/(τ*np.sqrt(2*(sigma_R**2*dl**2+τ**2)))
        beta_Nume_raw = 1/(sigma_R**2*np.sqrt(2*pi)) * np.exp(-(r**2)/(2*τ**2))*(sigma_R**2*τ**2)/(sigma_R**2*dl**2+τ**2) + 1/(2*np.sqrt(pi))*np.exp(-r**2/(2*(dl**2*sigma_R**2+τ**2))) * (sigma_R*dl*r*τ)/(dl**2*sigma_R**2+τ**2)**(3/2) * (np.sqrt(pi)*( 1-erf(coeff_erf*r, dtype=float)) ) 
        # print("by hand： " + str(beta_Nume_raw[0:2]))
        # Nume_raw_int, _ = quad_vec(f = lambda h:   normal.pdf((r-dl*h)/τ) * rayleigh.pdf(h, 0, sigma_R), a=0, b=np.Infinity) 
        # print("機器算的： " + str(Nume_raw_int[0:2]))

        β = (p0 * beta_Nume_raw) / (p0 * beta_Nume_raw + (1-p0)*normal.pdf(r/τ) )

    return β



def check_if_identical_msgs(Paths, cs_decoded_tx_message, L,J,parityLengthVector,messageLengthvector):   
    msg_bits = extract_msg_bits(Paths,cs_decoded_tx_message, L,J,parityLengthVector,messageLengthvector)
    flag = (msg_bits == msg_bits[0]).all()    
    return flag



def extract_msg_bits(Paths,cs_decoded_tx_message, L,J,parityLengthVector,messageLengthvector):
    msg_bits = np.empty(shape=(0,0), dtype=int)
    L1 = Paths.shape[0]
    for i in range(L1):
        msg_bit=np.empty(shape=(0,0), dtype=int)
        path = Paths[i].reshape(1,-1)
        for j in range(path.shape[1]):
            msg_bit = np.hstack((msg_bit,cs_decoded_tx_message[path[0,j],J*j:J*j+messageLengthvector[j]].reshape(1,-1))) if msg_bit.size else cs_decoded_tx_message[path[0,j],J*(j):J*(j)+messageLengthvector[j]]
            msg_bit=msg_bit.reshape(1,-1)
        msg_bits = np.vstack((msg_bits,msg_bit)) if msg_bits.size else msg_bit           
    return msg_bits



def analyze_genie_metrics(decTempBETA, L, J, listSize, txBitsParitized, K):
    thisTimeGenie = 0
    decOutMsg = convert_sparse_to_bits(decTempBETA, L, J, listSize, ) 
    error_box = []
    for i in range(txBitsParitized.shape[0]): # Recall txBitsParitized shape (100,256)
        oneOutageSection = 0
        flag_i = 1
        num_not_match_i = 0
        for l in range(L):                
            tmp = np.equal(txBitsParitized[i, l*J: (l+1)*J ], 
                            decOutMsg[:, l*J: (l+1)*J ]).all(axis=1).any()
            if tmp != True:
                flag_i = 0
                num_not_match_i += 1
                if num_not_match_i == 1: 
                    oneOutageSection = l
        thisTimeGenie += flag_i
        if (flag_i ==0):
            error_box.append(num_not_match_i)
            if (num_not_match_i == 1): 
                print(" | Some one-outage message has that one-outage at section:"  + str(oneOutageSection))
    print(" | genie recovers " + str(thisTimeGenie) +" out of " + str(K))
    print(" | How many sections do they lose? " + str(error_box))




def get_sig_values_and_positions(decTempBETA, L, J, listSize):
    decBETA_erase_small_values = postprocess_evenlS(decTempBETA,L,J,listSize)
    decBetaSignificants = np.zeros((L, listSize) )
    decBetaSignificantsPos = np.zeros((L, listSize), dtype=int )
    for l in np.arange(L):
        for n in np.arange(listSize):
            decBeta_l = decBETA_erase_small_values[l*2**J : (l+1)*2**J]
            decBetaSignificants[l] = decBeta_l[decBeta_l!=0]
            decBetaSignificantsPos[l] = [pos for decBeta_l_element, pos in zip(decBeta_l,np.arange(len(decBeta_l))) if decBeta_l_element!=0]
    decBetaSignificants = decBetaSignificants.transpose() # shape is (listSize, 16)
    decBetaSignificantsPos = decBetaSignificantsPos.transpose() # shape is (listSize, 16)
    return decBetaSignificants, decBetaSignificantsPos


def postprocess_evenlS(β1_dec, L, J, listSize):
    toGiveOut = np.array([])
    for i in range(L):
        # extract the information of i-th section
        A = β1_dec[i*2**J:(i+1)*2**J]
        # idx is the indices of the smallest (2**J - listSize) entries (fading values) in section i.
        idx = (A.reshape(2**J,)).argsort()[np.arange(2**J-listSize, dtype=int)]
        # B = np.setdiff1d(np.arange(2**J),idx)
        # temp = np.zeros(2**J)
        temp = A
        # idx是小值indices 所以都變成0
        temp[idx] = 0
        temp = temp.reshape(-1)

        toGiveOut = np.concatenate((toGiveOut,temp))
    return np.array(toGiveOut)


def convert_sparse_to_bits(cs_decoded_tx_message_sparse,L,J,listSize):
    cs_decoded_tx_message = np.zeros((listSize,L*J),dtype=int)
    for i in range(L):
        A = cs_decoded_tx_message_sparse[i*2**J:(i+1)*2**J]
        idx = (A.reshape(2**J,)).argsort()[np.arange(2**J-listSize)]
        B = np.setdiff1d(np.arange(2**J),idx)
        C = np.empty(shape=(0,0),dtype=int)
        for j in B:
            C = np.hstack((C,np.array([j],dtype=int))) if C.size else np.array([j],dtype=int)
        cs_decoded_tx_message[:,i*J:(i+1)*J]=np.array([list(np.binary_repr(int(x),J)) for x in C], dtype=int)    
    return cs_decoded_tx_message