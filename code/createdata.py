# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 16:32:30 2018

@author: Winry
"""

from sklearn.mixture import GMM
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data = pd.read_csv('')
n_components = np.arange(50, 210, 10)
models = [GMM(n, covariance_type='full', random_state=0)
          for n in n_components]
aics = [model.fit(data).aic(data) for model in models]
plt.plot(n_components, aics);

# 找到最低点 如果最低点为110 a = 110

a = 110

gmm = GMM(a, covariance_type='full', random_state=0)
gmm.fit(data)
print(gmm.converged_)
# 生成100个数据
data_new = gmm.sample(100, random_state=0)
data_new.shape