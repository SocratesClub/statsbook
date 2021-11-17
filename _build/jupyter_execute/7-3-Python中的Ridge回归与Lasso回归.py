#!/usr/bin/env python
# coding: utf-8

# # 第 7 章　统计学与机器学习
# 
# ## 第 3 节　Python 中的 Ridge 回归与 Lasso 回归
# 
# 
# 

# ### 2. 环境准备

# In[1]:


# 用于数值计算的库
import numpy as np
import pandas as pd
import scipy as sp
from scipy import stats

# 用于绘图的库
from matplotlib import pyplot as plt
import seaborn as sns
sns.set()

# 用于估计统计模型的库 (部分版本会报出警告信息)
import statsmodels.formula.api as smf
import statsmodels.api as sm

# 用于机器学习的库
from sklearn import linear_model

# 设置浮点数打印精度
get_ipython().run_line_magic('precision', '3')
# 在 Jupyter Notebook 里显示图形
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# 读入示例数据
X = pd.read_csv("7-3-1-large-data.csv")
X.head(3)


# ### 3. 实现：标准化

# In[3]:


# X_1 的均值
sp.mean(X.X_1)


# In[4]:


# 所有解释变量的均值
sp.mean(X, axis = 0).head(3)


# In[5]:


# 标准化
X -= sp.mean(X, axis = 0)
X /= sp.std(X, ddof = 1, axis = 0)


# In[6]:


# 检验
sp.mean(X, axis = 0).head(3).round(3)


# In[7]:


# 检验
sp.std(X, ddof = 1, axis = 0).head(3)


# ### 4. 定义响应变量

# In[8]:


# 定义响应变量

# 服从正态分布的噪声
np.random.seed(1)
noise =  sp.stats.norm.rvs(loc = 0, scale = 1, size = X.shape[0])

# 设正确的系数为 5, 定义响应变量
y =  X.X_1 * 5 + noise


# In[9]:


# 把响应变量和解释变量放在一起
large_data = pd.concat([pd.DataFrame({"y":y}), X], axis = 1)
# 绘制散点图
sns.jointplot(y = "y", x = "X_1", data = large_data,
              color = 'black')


# ### 5. 实现：普通最小二乘法

# In[10]:


lm_statsmodels = sm.OLS(endog = y, exog = X).fit()
lm_statsmodels.params.head(3)


# ### 6. 实现：使用 sklearn 实现线性回归

# In[11]:


# 指定模型的结构
lm_sklearn = linear_model.LinearRegression()
# 指定数据来源并估计模型
lm_sklearn.fit(X, y)
# 所估计的参数 (数组型)
lm_sklearn.coef_


# ### 7. 实现：Ridge 回归：惩罚项的影响

# In[12]:


# 生成 50 个 α
n_alphas = 50
ridge_alphas = np.logspace(-2, 0.7, n_alphas)


# In[13]:


# 参考
sp.log10(ridge_alphas)


# In[14]:


# 对不同的 α 值进行 Ridge 回归

# 存放已估计的回归系数的列表
ridge_coefs = []
# 使用 for 循环多次估计 Ridge 回归
for a in ridge_alphas:
    ridge = linear_model.Ridge(alpha = a, fit_intercept = False)
    ridge.fit(X, y)
    ridge_coefs.append(ridge.coef_)


# In[15]:


# 转换为数组
ridge_coefs = np.array(ridge_coefs)
ridge_coefs.shape


# In[16]:


# 参考
log_alphas = -sp.log10(ridge_alphas)
plt.plot(log_alphas, ridge_coefs[::,0], color = 'black')
plt.plot(log_alphas, ridge_coefs[::,1], color = 'black')

plt.xlim([min(log_alphas)-0.1, max(log_alphas) + 0.3])
plt.ylim([-8, 10.5])


# In[17]:


# 横轴为 -log10(α), 纵轴为系数的折线图
# 无需重复 100 次即可自动得到 100 条线

# 对 α 取对数
log_alphas = -sp.log10(ridge_alphas)
# 绘制曲线, 横轴为 -log10(α), 纵轴为系数
plt.plot(log_alphas, ridge_coefs, color = 'black')
# 标出解释变量 X_1 的系数
plt.text(max(log_alphas) + 0.1, np.array(ridge_coefs)[0,0], "X_1")
# X 轴的范围
plt.xlim([min(log_alphas) - 0.1, max(log_alphas) + 0.3])
# 轴标签
plt.title("Ridge")
plt.xlabel("- log10(alpha)")
plt.ylabel("Coefficients")


# ### 8. 实现：Ridge 回归：确定最佳正则化强度

# In[18]:


# 通过交叉验证法求最佳 α
ridge_best = linear_model.RidgeCV(
    cv = 10, alphas = ridge_alphas, fit_intercept = False)
ridge_best.fit(X, y) 

# 最佳的 -log10(α)
-sp.log10(ridge_best.alpha_)


# In[19]:


# 最佳 α
ridge_best.alpha_


# In[20]:


# 取最佳 α 时的回归系数
ridge_best.coef_


# ###  9. 实现：Lasso 回归：惩罚指标的影响

# In[21]:


# 对不同的 α 值进行 Lasso 回归
lasso_alphas, lasso_coefs, _ = linear_model.lasso_path(
    X, y, fit_intercept = False)


# In[23]:


# Lasso 回归的解路径图

# 对 α 取对数
log_alphas = -sp.log10(lasso_alphas)
# 绘制曲线, 横轴为 -log10(α), 纵轴为系数
plt.plot(log_alphas, lasso_coefs.T, color = 'black')
# 标出解释变量 X_1 的系数
plt.text(max(log_alphas) + 0.1, lasso_coefs[0, -1], "X_1")
# X 轴的范围
plt.xlim([min(log_alphas)-0.1, max(log_alphas) + 0.3])
# 轴标签
plt.title("Lasso")
plt.xlabel("- log10(alpha)")
plt.ylabel("Coefficients")


# ###  10. 实现：Lasso 回归：确定最佳正则化强度

# In[39]:


# 通过交叉验证法求最佳的 α
lasso_best = linear_model.LassoCV(
    cv = 10, alphas = lasso_alphas, fit_intercept = False)
lasso_best.fit(X, y)

# 最佳的 -log(α)
-sp.log10(lasso_best.alpha_)


# In[40]:


# 最佳的 α
lasso_best.alpha_


# In[41]:


# 取最佳的 α 时的回归系数
lasso_best.coef_


# In[ ]:




