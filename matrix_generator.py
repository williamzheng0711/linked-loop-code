import numpy as np
import binmatrix as BM
import random

def check_all_possiblility(num_row, num_col, trials):
    nr = num_row
    nc = num_col 
    i = 0
    while i < trials:
        A = np.random.randint(2, size=(nr, nc))
        checking = check_conditions(A, nr, nc)
        if checking: 
            print(A)
        i += 1

def check_conditions(A, num_row, num_col):
    binMat = BM.BinMatrix(m= A)
    if binMat.rank() < min(num_col, num_row):
        return False
    
    binMat2 = BM.BinMatrix(m = A[:,0:8])
    if binMat2.rank() < min(num_col, num_row):
        return False
    
    if num_col >= 4 and num_row>=4:
        for row in np.arange(0, num_row-3):
            for col in np.arange(0, num_col-3):
                B = A[row:row+3, col: col+3]
                if np.abs( 2*sum(sum(B)) - (B.shape[0])**2 ) > 3: return False

    return True


# check_all_possiblility(8,16,500000)


def which_columns_invertible(M):
    binMat = BM.BinMatrix(m= M)
    nc = binMat.c_len
    nr = binMat.r_len
    trials = 100
    for _ in range(trials):
        arr_list = np.sort(np.array(random.sample(range(0, 8 ),  nr)))
        binMat_thin = BM.BinMatrix(m= M[:,arr_list])
        if binMat_thin.rank() == nr:
            print(arr_list)
            print("----")
            print(binMat_thin.inv())
            return -1
 #                       0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13
#messageLens = np.array([9, 9, 9,10, 9, 9, 9, 9, 9, 9,10, 9, 9, 9])
# parityLens= np.array ([7, 7, 7, 6, 7, 7, 7, 7, 7, 7, 6, 7, 7, 7])

M =  np.array(
[[1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0],
 [1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0],
 [0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
 [1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0],
 [0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1],
 [1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0],
 [0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0],
 [1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1],],
            )
which_columns_invertible(M)