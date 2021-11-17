#!/usr/bin/env python
# coding: utf-8

# 
# # 第 5 章　正态线性模型
# 
# ## 第 1 节　含有单个连续型解释变量的模型（一元回归）

# ### 1. 环境准备

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

# 设置浮点数打印精度
get_ipython().run_line_magic('precision', '3')
# 在 Jupyter Notebook 里显示图形
get_ipython().run_line_magic('matplotlib', 'inline')


# ### 2. 实现：读入数据并绘制其图形

# In[2]:


# 读入数据
beer = pd.read_csv("5-1-1-beer.csv")
print(beer.head())


# In[3]:


# 绘制图像
sns.jointplot(x = "temperature", y = "beer", 
              data = beer, color = 'black')


# ### 4. 实现：使用 statsmodels 实现模型化

# In[4]:


# 建模
lm_model = smf.ols(formula = "beer ~ temperature", 
                   data = beer).fit()


# ### 5. 实现：打印估计结果并检验系数

# In[5]:


# 打印估计的结果
lm_model.summary()


# ### 7. 实现：使用 AIC 进行模型选择

# In[6]:


# 空模型
null_model = smf.ols("beer ~ 1", data = beer).fit()


# In[7]:


# 空模型的 AIC
null_model.aic


# In[8]:


# 含有解释变量的模型的 AIC
lm_model.aic


# In[9]:


# 对数似然度
lm_model.llf


# In[10]:


# 解释变量的个数
lm_model.df_model


# In[11]:


# AIC
-2*(lm_model.llf - (lm_model.df_model + 1))


# ### 9. 实现：用 seaborn 绘制回归直线

# In[12]:


sns.lmplot(x = "temperature", y = "beer", data = beer,
           scatter_kws = {"color": "black"},
           line_kws    = {"color": "black"})


# ### 10. 实现：使用模型进行预测

# In[13]:


# 拟合值
lm_model.predict()


# In[14]:


# 预测
lm_model.predict(pd.DataFrame({"temperature":[0]}))


# In[15]:


# 气温为 0 度时的预测值等于截距
lm_model.params


# In[16]:


# 预测
lm_model.predict(pd.DataFrame({"temperature":[20]}))


# In[17]:


# 不使用 predict 函数进行预测
beta0 = lm_model.params[0]
beta1 = lm_model.params[1]
temperature = 20

beta0 + beta1 * temperature


# ### 11. 实现：获取残差

# In[18]:


# 获得残差
resid = lm_model.resid
resid.head(3)


# In[19]:


# 计算拟合值
y_hat = beta0 + beta1 * beer.temperature
y_hat.head(3)


# In[20]:


# 获得拟合值
lm_model.fittedvalues.head(3)


# In[21]:


# 手动计算残差
(beer.beer - y_hat).head(3)


# ### 13. 实现：决定系数

# In[22]:


# 决定系数
mu = sp.mean(beer.beer)
y = beer.beer
yhat = lm_model.predict()

sp.sum((yhat - mu)**2) / sp.sum((y - mu)**2)


# In[23]:


lm_model.rsquared


# In[24]:


sp.sum((yhat - mu)**2) + sum(resid**2)


# In[25]:


sp.sum((y - mu)**2)


# In[26]:


1 - sp.sum(resid**2) / sp.sum((y - mu)**2)


# ### 15. 实现：修正决定系数

# In[27]:


n = len(beer.beer)
s = 1
1 - ((sp.sum(resid**2) / (n - s - 1)) / 
    (sp.sum((y - mu)**2) / (n - 1)))


# In[28]:


lm_model.rsquared_adj


# ### 16. 实现：残差的直方图和散点图

# In[29]:


# 残差的直方图
sns.distplot(resid, color = 'black')


# In[30]:


# 残差的散点图
sns.jointplot(lm_model.fittedvalues, resid, 
              joint_kws={"color": "black"}, 
              marginal_kws={"color": "black"})


# ### 18. 实现：分位图

# In[31]:


# 分位图
fig = sm.qqplot(resid, line = "s")


# In[32]:


# 递增排列
resid_sort = resid.sort_values()
resid_sort.head()


# In[33]:


# 最小的数据所在位置
1 / 31


# In[34]:


# 按样本容量变换为 0 到 1 的范围, 得到理论累积概率
# 
nobs = len(resid_sort)
cdf = np.arange(1, nobs + 1) / (nobs + 1)
cdf


# In[35]:


# 累积概率对应的百分位数
ppf = stats.norm.ppf(cdf)
ppf


# In[36]:


# 参考: 横轴为理论分位数, 纵轴为已排序的实际数据, 绘出的散点图就是分位图
fig = sm.qqplot(resid, line = "s")

plt.plot(stats.norm.ppf(cdf), resid_sort, "o", color = "black")


# ### 19. 根据 summary 函数的输出分析残差

# In[37]:


# 打印估计的结果
lm_model.summary()

