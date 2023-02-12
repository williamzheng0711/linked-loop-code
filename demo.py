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

# p_e = [1/10, 1/25, 1/50, 1/100, 1/200]
phase1_true = np.array([ 16, 59, 80, 86, 95])
phase1_false= np.array([0, 0, 0, 0, 0])
phase2_true = np.array([28, 25, 14, 9, 5])
phase2_false= np.array([5, 1, 0, 1, 0])
p_es = np.array(["1/10", "1/25", "1/50", "1/100", "1/200"])
pyplot.axhline(y = 100, color = 'black', linestyle = 'dashed')
pyplot.bar(p_es, phase1_true)
pyplot.bar(p_es, phase2_true, bottom = phase1_true)
pyplot.bar(p_es, phase1_false, bottom = phase1_true + phase2_true)
pyplot.bar(p_es, phase2_false, bottom = phase1_true+phase1_false+phase2_true)


pyplot.legend([ "K=100","Ph.1 True", "Ph.2 True","Ph.1 False",  "Ph.2 False"])
pyplot.xlabel("p_e")
pyplot.ylim([0,105])
pyplot.show()