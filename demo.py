# from joblib import Parallel, delayed
# import numpy as np

# def process(i):
#     return np.array([i * i, 1]).reshape(1,-1)
    
# results = Parallel(n_jobs=-1)(delayed(process)(i) for i in range(10))
# print(np.array(results).reshape(-1,2))  # prints [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]

# for result in results:
#     print(sum(result).all())
#     print(result.all() != None)




# import numpy as np
# import scipy
# from scipy.stats import bernoulli

# p = 0.1
# r = bernoulli.rvs(p, size=1000)
# print(r)



# import numpy as np
# import matplotlib.pyplot as pyplot


# phase1_llc_true = np.array([20, 36, 68, 100])
# phase1_llc_false= np.array([0,  0,  0, 0])
# phase2_llc_true = np.array([36, 38, 22, 0])
# phase2_llc_false= np.array([0,  2,  0, 0])
# phase1_tc_true =  np.array([20, 36, 68, 100])
# phase1_tc_false=  np.array([0, 0, 0, 0])
# p_es = np.array(["0.1", "0.05", "0.025", "0"])
# x_axis = np.arange(len(p_es))

# pyplot.axhline(y = 100, color = 'black', linestyle = 'dashed')
# pyplot.bar(x_axis-0.15, phase1_llc_true,  0.2)
# pyplot.bar(x_axis-0.15, phase2_llc_true,  0.2, bottom = phase1_llc_true)
# pyplot.bar(x_axis-0.15, phase1_llc_false, 0.2, bottom = phase1_llc_true + phase2_llc_true)
# pyplot.bar(x_axis-0.15, phase2_llc_false, 0.2, bottom = phase1_llc_true + phase2_llc_true + phase1_llc_false)

# pyplot.bar(x_axis+0.15, phase1_tc_true,  0.2)
# pyplot.bar(x_axis+0.15, phase1_tc_false, 0.2,  bottom = phase1_tc_true)

# pyplot.xticks(x_axis, p_es)
# pyplot.legend([ "K=100","Ph.1 True (LLC)", "Ph.2 True (LLC)","Ph.1 False (LLC)",  "Ph.2 False (LLC)", "Ph.1 True (TreeCode)", "Ph.1 False (TreeCode)"])
# pyplot.xlabel("p_e")
# pyplot.ylim([0,110])
# pyplot.title("K= 100 Linked-loop code and Tree code")
# pyplot.show()



# phase1_llc_true = np.array([28, 59, 107, 150])
# phase1_llc_false= np.array([0,  0,  0, 0])
# phase2_llc_true = np.array([37, 56, 30, 0])
# phase2_llc_false= np.array([4,  3,  0, 0])
# phase1_tc_true  = np.array([28, 59, 108, 150])
# phase1_tc_false = np.array([1,  0,  0, 0])

# pyplot.axhline(y = 150, color = 'black', linestyle = 'dashed')
# pyplot.bar(x_axis-0.15, phase1_llc_true,  0.2)
# pyplot.bar(x_axis-0.15, phase2_llc_true,  0.2, bottom = phase1_llc_true)
# pyplot.bar(x_axis-0.15, phase1_llc_false, 0.2, bottom = phase1_llc_true + phase2_llc_true)
# pyplot.bar(x_axis-0.15, phase2_llc_false, 0.2, bottom = phase1_llc_true + phase2_llc_true + phase1_llc_false)

# pyplot.bar(x_axis+0.15, phase1_tc_true,  0.2)
# pyplot.bar(x_axis+0.15, phase1_tc_false, 0.2,  bottom = phase1_tc_true)

# pyplot.xticks(x_axis, p_es)
# pyplot.legend([ "K=150","Ph.1 True (LLC)", "Ph.2 True (LLC)","Ph.1 False (LLC)",  "Ph.2 False (LLC)", "Ph.1 True (TreeCode)", "Ph.1 False (TreeCode)"])
# pyplot.xlabel("p_e")
# pyplot.ylim([0,160])
# pyplot.title("K= 150 Linked-loop code and Tree code")
# pyplot.show()










# import numpy as np
# from scipy.stats import bernoulli

# np.random.seed(2)
# applyErrs = np.where(bernoulli.rvs(0.1, size=100))[0]
# print(applyErrs)



import numpy as np

# create a sample 2D array
arr = np.array([[1, -1, 3], 
                [1, 0, -1], 
                [7, -1, 3]])

# count the number of -1 values in each column
counts = np.count_nonzero(arr == -1, axis=0)

# print(counts)

# chosenRoot = 5
# L = 16
# a = np.mod(np.arange(chosenRoot, chosenRoot+L),L)
# print(a)
# b = a[np.mod(np.arange(L-chosenRoot, 2*L-chosenRoot),L)]
# print(b)

# dummy = np.array([[2,2,2],
#                   [1,2,1],
#                   [1,2,4],
#                   [5,36,6]])
# # dummy2 = np.array([[1,1,3,4,5,6],
# #                   [3,3,3,3,3,3],
# #                   [3,3,3,3,3,3],])

# # aaa = np.unique(dummy2, axis=0)
# # print(aaa)

# row_to_find = np.array([1,2])
# matches = (dummy[:,0:2] ==row_to_find).all(axis=1)
# matching_indices = np.where(matches)[0]
# print(matching_indices)

# print(np.binary_repr(-1, width=16))



import numpy as np 

# tx_temp = np.array([[-1,2,2],
#                     [1,2,1],
#                     [1,2,4],
#                     [1,2,4],
#                     [5,36,6]])

# print(np.unique(tx_temp, axis=0))


# print( tx_temp[:,1].__contains__(36) )


from scipy.stats import bernoulli

np.random.seed(seed=2)
a = bernoulli.rvs(0.5, size=10)
print(a)


aa = [1,2,3,4,1]
b = np.unique(aa)

print(aa,b)



