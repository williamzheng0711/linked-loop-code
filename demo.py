# from joblib import Parallel, delayed
# import numpy as np

# def process(i):
#     return np.array([i * i, 1]).reshape(1,-1)
    
# results = Parallel(n_jobs=-1)(delayed(process)(i) for i in range(10))
# print(np.array(results).reshape(-1,2))  # prints [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]

# for result in results:
#     print(sum(result).all())
#     print(result.all() != None)


import numpy as np
import scipy
from scipy.stats import bernoulli

p = 0.1
r = bernoulli.rvs(p, size=1000)
print(r)
