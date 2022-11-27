import numpy as np


def check_all_possiblility(dim, trials):
    N = dim 
    i = 0
    while i < trials:
        A = np.random.randint(2, size=(N,N))
        checking = check_conditions(A)
        if checking: 
            print(A)
        i += 1

def check_conditions(A):
    if np.abs( 2*sum(sum(A)) - (A.shape[0])**2 ) > 1: return False
    if int(np.linalg.det(A) % 2) == 0: return False
    
    N = A.shape[0]
    if N >= 4:
        for row in np.arange(0, N-3):
            for col in np.arange(0, N-3):
                B = A[row:row+3, col: col+3]
                if np.abs( 2*sum(sum(B)) - (B.shape[0])**2 ) > 1: return False

    return True


check_all_possiblility(8,10000000)