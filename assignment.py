# !pip install -q glmnet_py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import glmnet_python
from glmnet import glmnet
from glmnetCoef import glmnetCoef
from glmnetPlot import glmnetPlot
from glmnetPredict import glmnetPredict
from glmnetPrint import glmnetPrint
from cvglmnet import cvglmnet
from cvglmnetPlot import cvglmnetPlot
from cvglmnetPredict import cvglmnetPredict

# データロード
df = pd.read_csv('QuickStartExample.csv', index_col = 0)
x = df.iloc[:, 0:20].values
y = df.iloc[:, 20].values

# Correlation
pd.set_option('display.max_columns', 21)
pd.set_option('display.max_rows', 21)
df.corr().style.background_gradient(cmap='cool')

# 解パス図
fit = glmnet(x = x, y = y)
glmnetPlot(fit, xvar = 'lambda', label = True);

# RMSE
np.random.seed(0)

k = 10 # K-fold cross-validation
foldid = np.random.choice(k, size = y.shape[0], replace = True)
cvfit = cvglmnet(x = x, y = y, foldid = foldid)

fig = plt.figure()
ax = fig.add_subplot(111)
cvglmnetPlot(cvfit)
plt.show()

# predict
y_hat = cvglmnetPredict(cvfit, newx = x, s = 'lambda_min')
fig = plt.figure()
ax = fig.add_subplot(111)

ax.scatter(y, y_hat)
ax.plot([-8, 8], [-8, 8], color = 'red')
ax.set_xlabel('y')
ax.set_ylabel('y_hat')

plt.show()