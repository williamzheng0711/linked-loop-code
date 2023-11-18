import numpy as np
import binmatrix as BM
import random

def check_all_possiblility(num_row, num_col, trials):
    nr = num_row
    nc = num_col 
    i = 0
    count = 0

    mat_array = []

    while i < trials:
        A = np.random.randint(2, size=(nr, nc))
        checking = check_conditions(A, nr, nc, offset=5)
        if checking:
            count += 1
            if (count % 2 == 0):
                print(np.hstack((mat_array[-1], A)))
            mat_array.append(A)
        i += 1

def check_conditions(A, num_row, num_col, offset):
    # It must be full rank
    binMat = BM.BinMatrix(m= A)
    if binMat.rank() < min(num_col, num_row):
        return False
    
    # The first square sub-matrix must be full rank
    binMat2 = BM.BinMatrix(m = A[:,0 : num_row])
    if binMat2.rank() < min(num_col, num_row):
        return False

    if num_col >= 4 and num_row>=4:
        for row in np.arange(0, num_row-3):
            for col in np.arange(0, num_col-3):
                B = A[row:row+3, col: col+3]
                if np.abs( 2*sum(sum(B)) - (B.shape[0])**2 ) > offset: return False

    return True


# check_all_possiblility(8, 8,20000000)


def which_columns_invertible(M_arr):
    for i in range(len(M_arr) ):
        M = M_arr[i]
        binMat = BM.BinMatrix(m= M)
        nc = binMat.c_len
        nr = binMat.r_len
        binMat_thin = BM.BinMatrix(m= M[:,range(0, nr)])
        if binMat_thin.rank() == nr:
            # print("----")
            print(binMat_thin.inv())
            # return -1


M =  np.array(
[


]
            )
which_columns_invertible(M)