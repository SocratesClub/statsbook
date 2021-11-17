#!/usr/bin/env python
# coding: utf-8

# # 第 5 节　泊松回归
# 
# ## 第 6 章　广义线性模型｜用 Python 动手学统计学
# 
# 

# ### 4. 环境准备

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


# In[2]:


# 读取数据
beer = pd.read_csv("6-5-1-poisson-regression.csv")
print(beer.head(3))


# ### 5. 实现：泊松回归

# In[3]:


# 建模
mod_pois = smf.glm("beer_number ~ temperature", beer, 
                   family=sm.families.Poisson()).fit()
mod_pois.summary()


# ### 6. 实现：模型选择

# In[4]:


# 空模型
mod_pois_null = smf.glm(
    "beer_number ~ 1", data = beer, 
    family=sm.families.Poisson()).fit()


# In[5]:


# 对比 AIC
print("空模型　：", mod_pois_null.aic.round(3))
print("气温模型：", mod_pois.aic.round(3))


# ### 7. 实现：回归曲线

# In[6]:


# 绘制回归曲线

# 计算预测值
x_plot = np.arange(0, 37)
pred = mod_pois.predict(
    pd.DataFrame({"temperature": x_plot}))

# 不含默认回归直线的 lmplot
sns.lmplot(y="beer_number", x = "temperature", 
           data = beer, fit_reg = False,
          scatter_kws = {"color":"black"})
# 绘出回归曲线
plt.plot(x_plot, pred, color="black")


# ### 8. 回归系数的含义

# In[7]:


# 气温为 1 度时销售数量的期望
exp_val_1 = pd.DataFrame({"temperature": [1]})
pred_1 = mod_pois.predict(exp_val_1)

# 气温为 2 度时销售数量的期望
exp_val_2 = pd.DataFrame({"temperature": [2]})
pred_2 = mod_pois.predict(exp_val_2)

# 气温每升高 1 度, 销量变为多少倍
pred_2 / pred_1


# In[8]:


# e 的指数为回归系数
sp.exp(mod_pois.params["temperature"])

