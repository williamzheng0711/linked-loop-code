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

phase1_true = np.array([109, 125])
phase1_false= np.array([0,  0])
phase2_true = np.array([26, 15])
phase2_false= np.array([4,  5])
p_es = np.array(["0.02", "0.01"])
pyplot.axhline(y = 150, color = 'black', linestyle = 'dashed')
pyplot.bar(p_es, phase1_true)
pyplot.bar(p_es, phase2_true, bottom = phase1_true)
pyplot.bar(p_es, phase1_false, bottom = phase1_true + phase2_true)
pyplot.bar(p_es, phase2_false, bottom = phase1_true+phase1_false+phase2_true)


pyplot.legend([ "K=100","Ph.1 True", "Ph.2 True","Ph.1 False",  "Ph.2 False"])
pyplot.xlabel("p_e")
pyplot.ylim([0,155])
pyplot.title("K= 150")
pyplot.show()