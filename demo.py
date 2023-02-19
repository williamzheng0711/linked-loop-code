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



import numpy as np
import matplotlib.pyplot as pyplot


phase1_llc_true = np.array([15, 37, 63])
phase1_llc_false= np.array([0,  0,  0])
phase2_llc_true = np.array([31, 39, 31])
phase2_llc_false= np.array([0,  3,  0])
phase1_tc_true = np.array([14, 37, 63])
phase1_tc_false= np.array([1,  1,  3])
p_es = np.array(["0.1", "0.05", "0.025"])
x_axis = np.arange(len(p_es))

pyplot.axhline(y = 100, color = 'black', linestyle = 'dashed')
pyplot.bar(x_axis-0.15, phase1_llc_true,  0.2)
pyplot.bar(x_axis-0.15, phase2_llc_true,  0.2, bottom = phase1_llc_true)
pyplot.bar(x_axis-0.15, phase1_llc_false, 0.2, bottom = phase1_llc_true + phase2_llc_true)
pyplot.bar(x_axis-0.15, phase2_llc_false, 0.2, bottom = phase1_llc_true + phase2_llc_true + phase1_llc_false)

pyplot.bar(x_axis+0.15, phase1_tc_true,  0.2)
pyplot.bar(x_axis+0.15, phase1_tc_false, 0.2,  bottom = phase1_tc_true)

pyplot.xticks(x_axis, p_es)
pyplot.legend([ "K=100","Ph.1 True (LLC)", "Ph.2 True (LLC)","Ph.1 False (LLC)",  "Ph.2 False (LLC)", "Ph.1 True (TreeCode)", "Ph.1 False (TreeCode)"])
pyplot.xlabel("p_e")
pyplot.ylim([0,110])
pyplot.title("K= 100 Linked-loop code and Tree code")
pyplot.show()



phase1_llc_true = np.array([28, 65, 96])
phase1_llc_false= np.array([0,  0,  0])
phase2_llc_true = np.array([48, 49, 43])
phase2_llc_false= np.array([4,  8,  2])
phase1_tc_true  = np.array([28, 65, 96])
phase1_tc_false = np.array([3,  8,  6])
p_es = np.array(["0.1", "0.05", "0.025"])
x_axis = np.arange(len(p_es))

pyplot.axhline(y = 150, color = 'black', linestyle = 'dashed')
pyplot.bar(x_axis-0.15, phase1_llc_true,  0.2)
pyplot.bar(x_axis-0.15, phase2_llc_true,  0.2, bottom = phase1_llc_true)
pyplot.bar(x_axis-0.15, phase1_llc_false, 0.2, bottom = phase1_llc_true + phase2_llc_true)
pyplot.bar(x_axis-0.15, phase2_llc_false, 0.2, bottom = phase1_llc_true + phase2_llc_true + phase1_llc_false)

pyplot.bar(x_axis+0.15, phase1_tc_true,  0.2)
pyplot.bar(x_axis+0.15, phase1_tc_false, 0.2,  bottom = phase1_tc_true)

pyplot.xticks(x_axis, p_es)
pyplot.legend([ "K=150","Ph.1 True (LLC)", "Ph.2 True (LLC)","Ph.1 False (LLC)",  "Ph.2 False (LLC)", "Ph.1 True (TreeCode)", "Ph.1 False (TreeCode)"])
pyplot.xlabel("p_e")
pyplot.ylim([0,160])
pyplot.title("K= 150 Linked-loop code and Tree code")
pyplot.show()










# import numpy as np
# from scipy.stats import bernoulli

# np.random.seed(2)
# applyErrs = np.where(bernoulli.rvs(0.1, size=100))[0]
# print(applyErrs)