from numpy import pi, sqrt
import numpy as np
from scipy.special import erf
from scipy.stats import norm as normal
from scipy.stats import rice, rayleigh
from scipy.integrate import quad_vec
from sklearn.linear_model import OrthogonalMatchingPursuit
import matplotlib.pyplot as pyplot
import timeit
from scipy.linalg import hadamard
# from stolen_things import *


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



# parity and info bits get mixed
# P is sum( paritylenthVector ) 

def generate_parity_distribution(identity=False):
    parityDistribution = np.zeros((16,16),dtype=int)
    parityDistribution[0][1] = 7; parityDistribution[0][2] = 4; parityDistribution[0][3] = 3; parityDistribution[0][4] = 2; 
    parityDistribution[1][2] = 3; parityDistribution[1][3] = 2; parityDistribution[1][4] = 2; parityDistribution[1][5] = 2; 
    parityDistribution[2][3] = 3; parityDistribution[2][4] = 2; parityDistribution[2][5] = 2; parityDistribution[2][6] = 2; 
    
    for i in np.arange(3,12,1):
        for j in np.arange(i + 1, i + 5, 1):
            parityDistribution[i][j] = 2

    parityDistribution[12][13] = 3; parityDistribution[12][14] = 2; parityDistribution[12][15] = 3
    parityDistribution[13][14] = 3; parityDistribution[13][15] = 4;  
    parityDistribution[14][15] = 7

    if identity!= True:
        useWhichMatrix = np.zeros((16,16),dtype=int)
        for row in np.arange(0,16):
            for col in np.arange(0, 16):
                if parityDistribution[row][col]!=0:
                    dim = parityDistribution[row][col]
                    choices = matrix_repo(dim=dim)
                    # print(choices)
                    useWhichMatrix[row][col] = np.random.randint(low=0, high=len(choices))

    elif identity == True:
        useWhichMatrix = np.zeros((16,16),dtype=int)
        for row in np.arange(0,16):
            for col in np.arange(0, 16):
                if parityDistribution[row][col]!=0:
                    dim = parityDistribution[row][col]
                    choices = matrix_repo(dim=dim)
                    # print(choices)
                    useWhichMatrix[row][col] = 0

    return parityDistribution, useWhichMatrix






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


# P : Total number of parity check bits
# Ml: Total number of information bits
def Tree_error_correct_encode(tx_message,K,L,J,P,Ml,messageLengthVector,parityLengthVector, parityDistribution, useWhichMatrix):
    encoded_tx_message = np.zeros((K,Ml+P),dtype=int)
    # plug in the info bits for each section
    encoded_tx_message[:,0:messageLengthVector[0]] = tx_message[:,0:messageLengthVector[0]]
    for i in range(1,L):
        encoded_tx_message[:,i*J:i*J+messageLengthVector[i]] = tx_message[:,np.sum(messageLengthVector[0:i]):np.sum(messageLengthVector[0:i+1])]
    
    for i in np.arange(0,L,1):
        parityDistRow_i = np.nonzero(parityDistribution[i])[0]
        # print(parityDistRow_i)
        for j in parityDistRow_i:
            # when i=0, j will be 1 2 3 4
            # j 是要写入东西的section i 是东西的来源section
            # if (i==0 and j == 1):
            # print("---")
            # print(j*J+messageLengthVector[j]+sum(parityDistribution[0:i,j]))
            # print(j*J+messageLengthVector[j]+sum(parityDistribution[0:i+1,j]))
            # print( sum(messageLengthVector[0:i])+sum(parityDistribution[i,0:j]) )
            # print( sum(messageLengthVector[0:i])+sum(parityDistribution[i,0:j+1]))
            # print(i,j)
            if useWhichMatrix != [] :
                encoded_tx_message[:,j*J+messageLengthVector[j]+sum(parityDistribution[0:i,j]) :    j*J+messageLengthVector[j]+sum(parityDistribution[0:i+1,j])] = \
                    (np.matmul(np.array(matrix_repo(parityDistribution[i][j])[useWhichMatrix[i][j]]),(tx_message[:, sum(messageLengthVector[0:i])+sum(parityDistribution[i,0:j]) : sum(messageLengthVector[0:i])+sum(parityDistribution[i,0:j+1])]).transpose() )).transpose() % 2

            else: 
                encoded_tx_message[:,j*J+messageLengthVector[j]+sum(parityDistribution[0:i,j]) :    j*J+messageLengthVector[j]+sum(parityDistribution[0:i+1,j])] = \
                    tx_message[:, sum(messageLengthVector[0:i])+sum(parityDistribution[i,0:j]) : sum(messageLengthVector[0:i])+sum(parityDistribution[i,0:j+1])]


    np.savetxt('encoded_message.csv', encoded_tx_message[0].reshape(16,16), delimiter=',', fmt='%d')
    # print(encoded_tx_message[0,0:16])

    return encoded_tx_message



def Tree_error_correct_encode_tb(tx_message,K,L,J,P,Ml,messageLengthVector,parityLengthVector, parityDistribution):
    encoded_tx_message = np.zeros((K,Ml+P),dtype=int)
    # plug in the info bits for each section
    encoded_tx_message[:,0:messageLengthVector[0]] = tx_message[:,0:messageLengthVector[0]]
    for i in range(1,L):
        encoded_tx_message[:,i*J:i*J+messageLengthVector[i]] = tx_message[:,np.sum(messageLengthVector[0:i]):np.sum(messageLengthVector[0:i+1])]
    
    for i in np.arange(0,L,1):
        parityDistRow_i = np.nonzero(parityDistribution[i])[0]
        # print(parityDistRow_i)
        for j in parityDistRow_i:
            # when i=0, j will be 1 2 3 4
            # j 是要写入东西的section i 是东西的来源section
            # if (i==0 and j == 1):
            # print("---")
            # print(j*J+messageLengthVector[j]+sum(parityDistribution[0:i,j]))
            # print(j*J+messageLengthVector[j]+sum(parityDistribution[0:i+1,j]))
            # print( sum(messageLengthVector[0:i])+sum(parityDistribution[i,0:j]) )
            # print( sum(messageLengthVector[0:i])+sum(parityDistribution[i,0:j+1]))
            # print(i,j)
            encoded_tx_message[:,j*J+messageLengthVector[j]+sum(parityDistribution[0:i,j]) :    j*J+messageLengthVector[j]+sum(parityDistribution[0:i+1,j])] = \
                tx_message[:, sum(messageLengthVector[0:i])+sum(parityDistribution[i,0:j]) : sum(messageLengthVector[0:i])+sum(parityDistribution[i,0:j+1])]

    np.savetxt('encoded_message.csv', encoded_tx_message[0].reshape(16,16), delimiter=',', fmt='%d')
    return encoded_tx_message





def generate_parity_matrix(L,messageLengthVector,parityLengthVector):
    # Generate a full matrix, use only the portion needed for tree code
    G = []
    for i in range(1,L):
        Gp = np.random.randint(2,size=(np.sum(messageLengthVector[0:i]),parityLengthVector[i])).tolist()
        G.append(Gp)
    # return np.asarray(G,)
    return np.asarray(G, dtype=object)

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

def convert_bits_to_sparse(encoded_tx_message,L,J,K):
    encoded_tx_message_sparse=np.zeros((L*2**J,1),dtype=float)
    for i in range(L):
        A = encoded_tx_message[:,i*J:(i+1)*J]
        B = A.dot(2**np.arange(J)[::-1]).reshape([K,1])
        np.add.at(encoded_tx_message_sparse, i*2**J+B, 1)        
    return encoded_tx_message_sparse

def convert_bits_to_sparse_Rician(encoded_tx_message,L,J,K, v_Rician, sigma_Rician):
    count = 0
    encoded_tx_message_sparse=np.zeros((L*2**J,1),dtype=float)
    fading_coefficients = rice.rvs(b= v_Rician/sigma_Rician, scale= sigma_Rician, loc =0, size=K)
    print(fading_coefficients)
    for i in range(L):
        A = encoded_tx_message[:,i*J:(i+1)*J]
        B = A.dot(2**np.arange(J)[::-1]).reshape([K,1])
        for k in range(K):
            encoded_tx_message_sparse[i*2**J+B[k]] += fading_coefficients[k]
            # np.add.at(encoded_tx_message_sparse, i*2**J+B[k], fading_coefficients[k])        
            count += 1
    
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



    

def amp_prior_art(y, σ_n, P, L, M, T, Ab, Az, p0, K):
    n = y.size
    β = np.zeros((L*M, 1))
    z = y
    Phat = n*P/L
    
    for t in range(T):
        
        τ = np.sqrt(np.sum(z**2)/n)
        # effective observation
        s = (np.sqrt(Phat)*β + Az(z)).astype(np.longdouble) 
        # denoiser
        β = (p0*np.exp(-(s-np.sqrt(Phat))**2/(2*τ**2)))/ (p0*np.exp(-(s-np.sqrt(Phat))**2/(2*τ**2)) + (1-p0)*np.exp(-s**2/(2*τ**2))).astype(float).reshape(-1, 1)
        # residual
        z = y - np.sqrt(Phat)*Ab(β) + (z/(n*τ**2)) * (Phat*np.sum(β) - Phat*np.sum(β**2))
        #print(t,τ)

    return β


def amp_prior_art_Rayleigh(y, σ_n, P, L, M, T, Ab, Az, p0, K, sigma_R, convertToBeta):
    n = y.size
    β = np.zeros((L*M, 1))
    z = y
    Phat = n*P/L
    dl = np.sqrt(Phat)

    # print("y is like: " + str(y) + " dl=" + str(np.sqrt(Phat)) )
    
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




def amp_prior_art_Rician(y, σ_n, P, L, M, T, Ab, Az, p0, K, v_Rician, sigma_Rician, convertToBeta):
    n = y.size
    β = np.zeros((L*M, 1))
    z = y
    Phat = n*P/L
    dl = np.sqrt(Phat)
    print("dl=" +str(dl))
    
    for t in range(T):
        print("-------------------Iter "+str(t) +" begins-------------------")

        # estimated s.e. of ζ
        τ = np.sqrt(np.sum(z**2)/n)

        # effective observation     r = d * hs + τζ
        r = (np.sqrt(Phat)*β + Az(z)).astype(np.longdouble) 
        print("r length is " + str(len(r)))

        print("r[0]=" + str(r[0]))
        print("tau=" + str(τ))

        # print(str(normal))
        # print(str(rice))

        # denoiser: β is E[hs|r]
        # if (t < T-1):
        # Rician distribution pdf: rice.pdf(h, v_Rician/sigma_Rician, 0, sigma_Rician) =  h/sigma_Rician**2 * np.exp(-(h**2 + v_Rician**2)/(2*sigma_Rician**2)) * i0(v_Rician/sigma_Rician**2 * h) 
        
        start = timeit.default_timer()
        Nume_int_part, _ = quad_vec(f = lambda h:   h*normal.pdf((r-dl*h)/τ) * rice.pdf(h, v_Rician/sigma_Rician, 0, sigma_Rician), a=0, b=np.Infinity) 
        stop = timeit.default_timer()
        print('Nume okay. Time: ' + str(stop - start))     

        start = timeit.default_timer()
        Deno_int_part, _ = quad_vec(f = lambda h:     normal.pdf((r-dl*h)/τ) * rice.pdf(h, v_Rician/sigma_Rician, 0, sigma_Rician), a=0, b=np.Infinity) 
        stop = timeit.default_timer()
        print('Deno okay. Time: ' + str(stop - start)) 

        # β is E[hs|r]
        β = p0 * Nume_int_part / ( p0 * Deno_int_part  +   (1-p0) * normal.pdf(r/τ) )                                     
        print("sum(beta) is: " + str(sum(β)))
    
        # residual
        # z = y - np.sqrt(Phat)*Ab(β) + (z/(n*τ**2)) * (Phat*np.sum(β) - Phat*np.sum((β)**2))
        int_part, _ =  quad_vec(f = lambda h:   h**2 * normal.pdf((r-dl*h)/τ) * rice.pdf(h, v_Rician/sigma_Rician, 0, sigma_Rician), a=0, b=np.Infinity) 
        # E_h2s2 = E[h^2 s^2 | r]
        E_h2s2 = p0 * int_part / ( p0 * Deno_int_part  +   (1-p0) * normal.pdf(r/τ) )     

        z = y - np.sqrt(Phat)*Ab(β) + (z/n) * (Phat/τ**2) * sum(  E_h2s2 - β**2  )


    if convertToBeta:
        pme_int_part, _ = quad_vec(f = lambda h:     normal.pdf((r-dl*h)/τ) *  rice.pdf(h, v_Rician/sigma_Rician, 0, sigma_Rician)      , a=0, b=np.Infinity,) 
        β = p0 * pme_int_part / ( p0 * pme_int_part + (1-p0) * normal.pdf(r/τ) )
    print(β)

    return β















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




def Tree_decoder_uninformative(cs_decoded_tx_message,G,L,J,B,parityLengthVector,messageLengthvector,listSize):
    tree_decoded_tx_message = np.empty(shape=(0,0))
    for i in range(listSize):
        Paths = np.array([[i]])
        for l in range(1,L):
            # Grab the parity generator matrix corresponding to this section
            G1 = G[l-1]
            new=np.empty( shape=(0,0))
            for j in range(Paths.shape[0]):
                Path=Paths[j].reshape(1,-1)
                # print("i=" + str(i) + " j=" + str(j) + " and Path is" + str(Path))
                # Compute the permissible parity check bits for the section
                Parity_computed = compute_permissible_parity(Path,cs_decoded_tx_message,G1,L,J,parityLengthVector,messageLengthvector)
                # print("Parity_computed is: " + str(Parity_computed) )
                for k in range(listSize):
                    # Verify parity constraints for the children of surviving path
                    index = parity_check(Parity_computed,Path,k,cs_decoded_tx_message,L,J,parityLengthVector,messageLengthvector)
                    # If parity constraints are satisfied, update the path
                    if index:
                        new = np.vstack((new,np.hstack((Path.reshape(1,-1),np.array([[k]]))))) if new.size else np.hstack((Path.reshape(1,-1),np.array([[k]])))
            Paths = new 
            
        if Paths.shape[0] >= 2:
            # If tree decoder outputs multiple paths for a root node, select the first one 
            flag = check_if_identical_msgs(Paths, cs_decoded_tx_message, L,J,parityLengthVector,messageLengthvector)
            if flag:
                tree_decoded_tx_message = np.vstack((tree_decoded_tx_message,extract_msg_bits(Paths[0].reshape(1,-1),cs_decoded_tx_message, L,J,parityLengthVector,messageLengthvector))) if tree_decoded_tx_message.size else extract_msg_bits(Paths[0].reshape(1,-1),cs_decoded_tx_message, L,J,parityLengthVector,messageLengthvector)
            else:
                # print("Path shape is" + str(Paths.shape))
                tree_decoded_tx_message = np.vstack((tree_decoded_tx_message,extract_msg_bits(Paths.reshape(Paths.shape[0],-1),cs_decoded_tx_message, L,J,parityLengthVector,messageLengthvector))) if tree_decoded_tx_message.size else extract_msg_bits(Paths.reshape(Paths.shape[0],-1),cs_decoded_tx_message, L,J,parityLengthVector,messageLengthvector)
        elif Paths.shape[0] == 1:
            tree_decoded_tx_message = np.vstack((tree_decoded_tx_message,extract_msg_bits(Paths.reshape(1,-1),cs_decoded_tx_message, L,J,parityLengthVector,messageLengthvector))) if tree_decoded_tx_message.size else extract_msg_bits(Paths.reshape(1,-1),cs_decoded_tx_message, L,J,parityLengthVector,messageLengthvector)
    
    return tree_decoded_tx_message





def compute_permissible_parity(Path,cs_decoded_tx_message,G1,L,J,parityLengthVector,messageLengthvector):
    msg_bits = extract_msg_bits(Path,cs_decoded_tx_message, L,J,parityLengthVector,messageLengthvector)
    Lpath = Path.shape[1]
    Parity_computed_integer = 0
    for i in range(Lpath):
        ParityBinary = np.mod(np.matmul(msg_bits[:,np.sum(messageLengthvector[0:i]):np.sum(messageLengthvector[0:i+1])],
                            G1[np.sum(messageLengthvector[0:i]):np.sum(messageLengthvector[0:i+1])]),2)
        ParityBinary=ParityBinary.reshape(1,-1)
        # Convert into decimal equivalent\n",
        ParityInteger1 = ParityBinary.dot(2**np.arange(ParityBinary.shape[1])[::-1])
        Parity_computed_integer = np.mod(Parity_computed_integer+ParityInteger1,2**parityLengthVector[Lpath])        
         
    Parity_computed = np.array([list(np.binary_repr(int(x),parityLengthVector[Lpath])) for x in Parity_computed_integer], dtype=int)
    return Parity_computed

def parity_check(Parity_computed,Path,k,cs_decoded_tx_message,L,J,parityLengthVector,messageLengthvector, parityDistribution, useWhichMatrix):
    index1 = 0
    index2 = 1
    Lpath = Path.shape[1]
    Parity = cs_decoded_tx_message[k,Lpath*J+messageLengthvector[Lpath]:(Lpath+1)*J]
    check_args = np.where(Parity_computed >=0)[0]
    if (np.sum(np.absolute(Parity_computed[check_args]-Parity[check_args])) == 0):
        index1 = 1
    
    if Lpath >= 13:
        cs_decoded_L_sections = np.ones((1,L*J), dtype=int)
        for ll in np.arange(Lpath):
            cs_decoded_L_sections[0][ll*J:(ll+1)*J] = cs_decoded_tx_message[Path[0][ll], ll*J:(ll+1)*J]

        for l in np.arange(12, Lpath):
            # check what sections are partly determined by l
            toCheckSections = np.nonzero(parityDistribution[l])[0] # is l = 13, then toCheckSections = [14, 15, 0, 1]
            # for each those sections, check if parity are same. 
            for section in toCheckSections: # section = 14, 15, 0, 1
                if section > l: continue # only section = 0, 1 will be executed
                
                # print('---- l='+str(l) +" and section=" + str(section) + "----")
                # print(section*J + messageLengthvector[section] + sum(parityDistribution[0:l,section]))
                # print(section*J + messageLengthvector[section] + sum(parityDistribution[0:l+1,section]))
                # print(l*J        + sum(parityDistribution[l,0:section]))
                # print(l*J        + sum(parityDistribution[l,0:section+1]))

                gen_mat = matrix_repo(parityDistribution[l][section])[useWhichMatrix[l][section]]
                # gen_binmat = BinMatrix(gen_mat)

                oldPart = cs_decoded_L_sections[0][section*J + messageLengthvector[section] + sum(parityDistribution[0:l,section]) : section*J + messageLengthvector[section] + sum(parityDistribution[0:l+1,section])].reshape(1,-1)[0]
                newPart = np.matmul(gen_mat ,cs_decoded_L_sections[0][      l*J                                + sum(parityDistribution[l,0:section]) :       l*J                                + sum(parityDistribution[l,0:section+1])] ).reshape(1,-1)[0]

                check_args_old = np.where(oldPart >=0)[0]
                # print("check_args_old" + str(check_args_old))
                checksum =  np.sum(np.absolute(  oldPart[check_args_old]-newPart[check_args_old]  ))
                # print("checksum = " + str(checksum))
                if  checksum != 0:
                    index2 = 0
                    return index2

    return index1 * index2

def check_if_identical_msgs(Paths, cs_decoded_tx_message, L,J,parityLengthVector,messageLengthvector):   
    msg_bits = extract_msg_bits(Paths,cs_decoded_tx_message, L,J,parityLengthVector,messageLengthvector)
    flag = (msg_bits == msg_bits[0]).all()    
    return flag

def extract_msg_bits(Paths,cs_decoded_tx_message, L,J,parityLengthVector,messageLengthvector):
    msg_bits = np.empty(shape=(0,0))
    L1 = Paths.shape[0]
    for i in range(L1):
        msg_bit=np.empty(shape=(0,0))
        path = Paths[i].reshape(1,-1)
        for j in range(path.shape[1]):
            msg_bit = np.hstack((msg_bit,cs_decoded_tx_message[path[0,j],J*j:J*j+messageLengthvector[j]].reshape(1,-1))) if msg_bit.size else cs_decoded_tx_message[path[0,j],J*(j):J*(j)+messageLengthvector[j]]
            msg_bit=msg_bit.reshape(1,-1)
        msg_bits = np.vstack((msg_bits,msg_bit)) if msg_bits.size else msg_bit           
    return msg_bits





def stitching_use_fading_and_tree_listSize( decBetaSignificants, decBetaSignificantsPos, K, G, J, P, Ml, messageLengthVector, parityLengthVector, L, listSize ):
    # shape of decBeta is 16 * 65536 ( L*2**J)

    rxOutercodes = np.array([])
    nowRecovered = 0
    alarmCondition = False
    decBetaSignificantsUpdate = decBetaSignificants
    decBetaSignificantsPosUpdate = decBetaSignificantsPos

    while (nowRecovered < K and alarmCondition == False):

        print("nowRecovered is " + str(nowRecovered))
        anOuterCode, _, decBetaSignificantsUpdate, decBetaSignificantsPosUpdate = extract_one_outer_code_listSize(decBetaSignificantsUpdate, decBetaSignificantsPosUpdate, K, G, J, P, Ml, messageLengthVector, parityLengthVector)
        if len(anOuterCode) != 0:
            answer = np.array([])
            for l in np.arange(L):
                a = np.binary_repr(anOuterCode[l], width=16)
                # answer.append( [int(n) for n in a] )
                answer = np.append(answer, [int(n) for n in a] ).reshape(1,-1)

            if nowRecovered == 0:
                rxOutercodes = answer
            else: 
                rxOutercodes = np.vstack((rxOutercodes, answer)) 
            nowRecovered += 1
        else: # if usedUp is true, then anOuterCode is null
            alarmCondition = True
    
    # rxOutercodes 最好是 (100, 256)的binary 可以直接進行 PUPE 的計算
    return rxOutercodes



# given some decBeta, this algorithm gives out a most probable outer code
def extract_one_outer_code_listSize(decBetaSignificants, decBetaSignificantsPos, K, G, J, P, Ml, messageLengthVector, parityLengthVector):
    usedUp = False 
    fadingValues = [] # entries are in range (0, infty)
    positionValues = [] # entries are in range (0, 65535) = (0, 2**J-1)

    L = decBetaSignificants.shape[0] # usually, L = 16
    # print("L = " + str(L))
    eachSectionArgs = np.zeros(L, dtype=int)

    while len(fadingValues) < L and usedUp == False: 
        # print("l = " + str(l))
        # print("positionValues = " + str(positionValues))

        lenFVsPrior = len(fadingValues)

        eachSectionArgs[ lenFVsPrior + 1 : ] = 0

        if (len(fadingValues)!=0):
            # 找最小的
            args = np.argsort( abs(decBetaSignificants[lenFVsPrior] - np.mean(fadingValues)) )
        else: 
            # 找最大的
            args = np.flip( np.argsort( decBetaSignificants[lenFVsPrior] ) )
        
        # if (eachSectionArgs[ len(fadingValues) ] == 0):
        #     arg_trials = eachSectionArgs[ len(fadingValues) ] 
        # else: 
        #     arg_trials = eachSectionArgs[ len(fadingValues) ] + 1

        arg_trials = eachSectionArgs[ len(fadingValues) ]

        parityConsistent = False  
        
        while parityConsistent == False and arg_trials < len(args) and usedUp == False:
            positionValues.append( decBetaSignificantsPos[ lenFVsPrior ][args[arg_trials]]   )
            fadingValues.append(      decBetaSignificants[ lenFVsPrior ][args[arg_trials]]   )
            if ( len(positionValues) >= 2 ):
                if positionValues[-1] >= 0:
                    parityConsistent = parity_check_part(K, G, L, J, P, Ml, messageLengthVector, parityLengthVector, positionValues)
                
                if (positionValues[-1] < 0 or parityConsistent == False):
                    positionValues.pop()
                    fadingValues.pop()
            else: # is len(positionValues) == 1, then true
                parityConsistent = True
            
            arg_trials += 1

        if len(fadingValues) == lenFVsPrior and len(fadingValues) != 0:
            # 我們失敗了, we retreat to last section
            positionValues.pop()
            fadingValues.pop()

        # elif len(fadingValues) == lenFVsPrior and len(fadingValues) == 0:
        #     usedUp = True
        #     break

        elif len(fadingValues) == lenFVsPrior and len(fadingValues) == 0:
            decBetaSignificants[   0][args[0]] = 0
            decBetaSignificantsPos[0][args[0]] = -1

        elif len(fadingValues) == lenFVsPrior + 1:
            # 成功, we succeed to find one proper candidate in this section
            eachSectionArgs[lenFVsPrior] = arg_trials

        if max(decBetaSignificants[   0]) == 0:
            usedUp == True
            break
        
    # anOuterCode = convert_positions_to_bits(positionValues)
    if usedUp == False and len(positionValues)==L:
        # for ll in np.arange(L):
        # Why in original code we only erase the root? 
        for ll in np.arange(0,1):
            index_l = np.where(decBetaSignificantsPos[ll] == positionValues[ll])
            # print("index of " + str(ll) + " is: " + str(index_l))
            decBetaSignificants[ll][index_l] = 0
            decBetaSignificantsPos[ll][index_l] = -1
        
        # print("fadingValues = " + str(fadingValues))
        # print("positionValues = " + str(positionValues))
        # print("eachSectionArgs = " + str(eachSectionArgs))

        return positionValues, usedUp, decBetaSignificants, decBetaSignificantsPos
    
    else: 
        return [], usedUp, decBetaSignificants, decBetaSignificantsPos
        

# 原先 tx_message 是 100 x 128的東西
def parity_check_part(K, G, L, J, P, Ml, messageLengthVector, parityLengthVector, positionValues):
    
    howManySectionInHand = len(positionValues)
    # print("Now howmany sections? " + str(howManySectionInHand))
    
    answer = np.array([])
    for l in np.arange( howManySectionInHand ):
        # print(positionValues)
        a = np.binary_repr(positionValues[l], width= J )
        answer = np.append(answer, [int(n) for n in a] ).reshape(1,-1)

    answer = np.array(answer).reshape(1,-1)
    # print("answer :" + str(answer))

    messageBySections = np.array([])
    # section ZERO is always okay to go
    currentSection = 0
    while (currentSection < howManySectionInHand ):
        messageBySections = np.append(messageBySections, answer[:, currentSection*messageLengthVector[0] : (currentSection+1)*messageLengthVector[0]-parityLengthVector[currentSection]])
        currentSection += 1

    messageBySections = messageBySections.reshape(1,-1)
    # print( "messageBySections :" + str(messageBySections) )
    
    # for i in range(1,L):
    i = howManySectionInHand - 1
    ParityInteger=np.zeros((1,1),dtype='int')
    G1=G[i-1]
    for j in range(1,i+1):
        ParityBinary = np.mod(
                            np.matmul(  messageBySections[ 0,np.sum(messageLengthVector[0:j-1]) : np.sum(messageLengthVector[0:j]) ],
                                        G1[ np.sum(messageLengthVector[0:j-1]) : np.sum(messageLengthVector[0:j]) ]
                                ),
                        2)
        # Convert into decimal equivalent\n",
        ParityBinary = ParityBinary.reshape(1,-1)
        # print("ParityBinary shape: " + str(ParityBinary.shape))
        ParityInteger1 = ParityBinary.dot(2**np.arange(ParityBinary.shape[1])[::-1]).reshape([1,1])
        ParityInteger = np.mod(ParityInteger+ParityInteger1,2**parityLengthVector[i])


    Parity = np.array([list(np.binary_repr(int(x),parityLengthVector[i])) for x in ParityInteger], dtype=int)
    # print("Parity via Calculation is = " + str(Parity))
    onHand = np.array(answer[:, answer.shape[1] - parityLengthVector[howManySectionInHand-1] : ], dtype=int).reshape(1,-1)
    # print("On hand is = " + str(onHand) )

    if np.array_equal(Parity, onHand):
        same = True
    else:
        same = False
    
    # print("checked " + str(same))

    return same
    



def postprocess(β1_dec, L, J, listSize):
    toGiveOut = np.array([])
    for i in range(L):
        A = β1_dec[i*2**J:(i+1)*2**J]
        idx = (A.reshape(2**J,)).argsort()[np.arange(2**J-listSize)]
        B = np.setdiff1d(np.arange(2**J),idx)
        temp = np.zeros(2**J)
        temp[B] = 1
        temp = temp.reshape(-1)

        toGiveOut = np.concatenate((toGiveOut,temp))
    return np.array(toGiveOut)


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


def postprocess_increasinglS(β1_dec, L, J, listSize):
    toGiveOut = np.array([])
    for i in range(L):
        # extract the information of i-th section
        A = β1_dec[i*2**J:(i+1)*2**J]
        # idx is the indices of the smallest (2**J - listSize) entries (fading values) in section i.
        idx = (A.reshape(2**J,)).argsort()[np.arange(2**J-listSize[i], dtype=int)]
        # B = np.setdiff1d(np.arange(2**J),idx)
        # temp = np.zeros(2**J)
        temp = A
        # idx是小值indices 所以都變成0
        temp[idx] = 0
        temp = temp.reshape(-1)

        toGiveOut = np.concatenate((toGiveOut,temp))
    return np.array(toGiveOut)




def Tree_decoder_uninformative_fading(decBetaSignificants, decBetaSignificantsPos, G,L,J,B,parityLengthVector,messageLengthvector,listSize):
    # decBetaSignificants size is (listSize, 16)
    cs_decoded_tx_message = np.zeros( (listSize, L*J) ) # (listSize, 256)
    for id_row in range(decBetaSignificantsPos.shape[0]):
        for id_col in range(decBetaSignificantsPos.shape[1]):
            a = np.binary_repr(decBetaSignificantsPos[id_row][id_col], width=J)
            # print("a = " + str(a))
            b = np.array([int(n) for n in a] ).reshape(1,-1)
            # print("b = " + str(b))
            cs_decoded_tx_message[ id_row, id_col*J: (id_col+1)*J ] = b[0, 0:J]

    listSizeOrder = np.argsort( decBetaSignificants[:,0] )
    # print("listSizeOrder is " + str(listSizeOrder))

    tree_decoded_tx_message = np.empty(shape=(0,0))
    for i in listSizeOrder:
        Paths = np.array([[i]])
        for l in range(1,L):
            # Grab the parity generator matrix corresponding to this section
            G1 = G[l-1]
            new=np.empty( shape=(0,0))
            for j in range(Paths.shape[0]):
                Path=Paths[j].reshape(1,-1)
                # print("i=" + str(i) + " j=" + str(j) + " and Path is" + str(Path))
                # Compute the permissible parity check bits for the section
                Parity_computed = compute_permissible_parity(Path,cs_decoded_tx_message,G1,L,J,parityLengthVector,messageLengthvector)
                # print("Parity_computed is: " + str(Parity_computed) )
                for k in range(listSize):
                    # Verify parity constraints for the children of surviving path
                    index = parity_check(Parity_computed,Path,k,cs_decoded_tx_message,L,J,parityLengthVector,messageLengthvector)
                    # If parity constraints are satisfied, update the path
                    if index:
                        new = np.vstack((new,np.hstack((Path.reshape(1,-1),np.array([[k]]))))) if new.size else np.hstack((Path.reshape(1,-1),np.array([[k]])))
            Paths = new 
            
        if Paths.shape[0] >= 2:
            # If tree decoder outputs multiple paths for a root node, select the first one 
            flag = check_if_identical_msgs(Paths, cs_decoded_tx_message, L,J,parityLengthVector,messageLengthvector)
            if flag:
                # print("Path[0] detail is " + str(Paths[0]))
                tree_decoded_tx_message = np.vstack((tree_decoded_tx_message,extract_msg_bits(Paths[0].reshape(1,-1),cs_decoded_tx_message, L,J,parityLengthVector,messageLengthvector))) if tree_decoded_tx_message.size else extract_msg_bits(Paths[0].reshape(1,-1),cs_decoded_tx_message, L,J,parityLengthVector,messageLengthvector)
            else:
                # print("Path shape is" + str(Paths.shape))
                # print("Path[0] detail is " + str(Paths[0]))
                optimalOne = 0
                pathVar = np.zeros((Paths.shape[0]))
                for whichPath in range(Paths.shape[0]):
                    fadingValues = []
                    for l in range(Paths.shape[1]):
                        # decBetaSignificantsPos size is (listSize, 16)s
                        fadingValues.append( decBetaSignificants[ Paths[whichPath][l] ][l] )
                    
                    # print("fadingValues = " + str(fadingValues))
                    demeanFading = fadingValues - np.mean(fadingValues)
                    # pathVar[whichPath] = np.linalg.norm(demeanFading, 1)
                    pathVar[whichPath] = np.var(fadingValues)

                optimalOne = np.argmin(pathVar)
                tree_decoded_tx_message = np.vstack((tree_decoded_tx_message,extract_msg_bits(Paths[optimalOne].reshape(1,-1),cs_decoded_tx_message, L,J,parityLengthVector,messageLengthvector))) if tree_decoded_tx_message.size else extract_msg_bits(Paths[optimalOne].reshape(1,-1),cs_decoded_tx_message, L,J,parityLengthVector,messageLengthvector)
                # tree_decoded_tx_message = np.vstack( (tree_decoded_tx_message,extract_msg_bits(Paths.reshape(Paths.shape[0],-1),cs_decoded_tx_message, L,J,parityLengthVector,messageLengthvector))) if tree_decoded_tx_message.size else extract_msg_bits(Paths.reshape(Paths.shape[0],-1),cs_decoded_tx_message, L,J,parityLengthVector,messageLengthvector)
        elif Paths.shape[0] == 1:
            tree_decoded_tx_message = np.vstack((tree_decoded_tx_message,extract_msg_bits(Paths.reshape(1,-1),cs_decoded_tx_message, L,J,parityLengthVector,messageLengthvector))) if tree_decoded_tx_message.size else extract_msg_bits(Paths.reshape(1,-1),cs_decoded_tx_message, L,J,parityLengthVector,messageLengthvector)
    
    return tree_decoded_tx_message




def Tree_decoder_uninformative_fading_growinglS(decBetaSignificants, decBetaSignificantsPos, G,L,J,B,parityLengthVector,messageLengthvector,listSize):
    # decBetaSignificants size is (16, "listSize")
    # Now "decBetaSignificants" are dictionary, not ordinary array
    cs_decoded_tx_message = np.ones( (listSize[-1], L*J) ) # (listSize[-1], 256)
    cs_decoded_tx_message = -1 * cs_decoded_tx_message

    for idx_l in range(L):
        for idx_ls in range(listSize[idx_l]):
            a = np.binary_repr(decBetaSignificantsPos[idx_l][idx_ls], width=J)
            # print("a = " + str(a))
            b = np.array([int(n) for n in a] ).reshape(1,-1)
            # print("b = " + str(b))
            cs_decoded_tx_message[idx_ls,   idx_l*J: (idx_l+1)*J ] = b[0, 0:J]

    listSizeOrder = np.argsort( decBetaSignificants[0] )
    # print("listSizeOrder is " + str(listSizeOrder))

    tree_decoded_tx_message = np.empty(shape=(0,0))
    for i in listSizeOrder:
        Paths = np.array([[i]])
        for l in range(1,L):
            # Grab the parity generator matrix corresponding to this section
            G1 = G[l-1]
            new=np.empty( shape=(0,0))
            for j in range(Paths.shape[0]):
                Path=Paths[j].reshape(1,-1)
                # print("i=" + str(i) + " j=" + str(j) + " and Path is" + str(Path))
                # Compute the permissible parity check bits for the section
                Parity_computed = compute_permissible_parity(Path,cs_decoded_tx_message,G1,L,J,parityLengthVector,messageLengthvector)
                # print("Parity_computed is: " + str(Parity_computed) )
                for k in range(listSize[l]):
                    # Verify parity constraints for the children of surviving path
                    index = parity_check(Parity_computed,Path,k,cs_decoded_tx_message,L,J,parityLengthVector,messageLengthvector)
                    # If parity constraints are satisfied, update the path
                    if index:
                        new = np.vstack((new,np.hstack((Path.reshape(1,-1),np.array([[k]]))))) if new.size else np.hstack((Path.reshape(1,-1),np.array([[k]])))
            Paths = new 
            
        if Paths.shape[0] >= 2:
            # If tree decoder outputs multiple paths for a root node, select the first one 
            flag = check_if_identical_msgs(Paths, cs_decoded_tx_message, L,J,parityLengthVector,messageLengthvector)
            if flag:
                # print("Path[0] detail is " + str(Paths[0]))
                tree_decoded_tx_message = np.vstack((tree_decoded_tx_message,extract_msg_bits(Paths[0].reshape(1,-1),cs_decoded_tx_message, L,J,parityLengthVector,messageLengthvector))) if tree_decoded_tx_message.size else extract_msg_bits(Paths[0].reshape(1,-1),cs_decoded_tx_message, L,J,parityLengthVector,messageLengthvector)
            else:
                # print("Path shape is" + str(Paths.shape))
                # print("Path[0] detail is " + str(Paths[0]))
                optimalOne = 0
                pathVar = np.zeros((Paths.shape[0]))
                for whichPath in range(Paths.shape[0]):
                    fadingValues = []
                    for l in range(Paths.shape[1]):
                        # decBetaSignificantsPos size is (16, "listSize")
                        fadingValues.append( decBetaSignificants[l][ Paths[whichPath][l] ] )
                    
                    pathVar[whichPath] = np.var(fadingValues)

                optimalOne = np.argmin(pathVar)
                tree_decoded_tx_message = np.vstack((tree_decoded_tx_message,extract_msg_bits(Paths[optimalOne].reshape(1,-1),cs_decoded_tx_message, L,J,parityLengthVector,messageLengthvector))) if tree_decoded_tx_message.size else extract_msg_bits(Paths[optimalOne].reshape(1,-1),cs_decoded_tx_message, L,J,parityLengthVector,messageLengthvector)
                # tree_decoded_tx_message = np.vstack( (tree_decoded_tx_message,extract_msg_bits(Paths.reshape(Paths.shape[0],-1),cs_decoded_tx_message, L,J,parityLengthVector,messageLengthvector))) if tree_decoded_tx_message.size else extract_msg_bits(Paths.reshape(Paths.shape[0],-1),cs_decoded_tx_message, L,J,parityLengthVector,messageLengthvector)
        elif Paths.shape[0] == 1:
            tree_decoded_tx_message = np.vstack((tree_decoded_tx_message,extract_msg_bits(Paths.reshape(1,-1),cs_decoded_tx_message, L,J,parityLengthVector,messageLengthvector))) if tree_decoded_tx_message.size else extract_msg_bits(Paths.reshape(1,-1),cs_decoded_tx_message, L,J,parityLengthVector,messageLengthvector)
    
    return tree_decoded_tx_message



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



