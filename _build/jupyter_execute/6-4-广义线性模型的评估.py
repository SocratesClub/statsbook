#!/usr/bin/env python
# coding: utf-8

# # 第 4 节　广义线性模型的评估
# 
# ## 第 6 章　广义线性模型｜用 Python 动手学统计学
# 
# 

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


# In[2]:


# 读取数据
test_result = pd.read_csv("6-3-1-logistic-regression.csv")

# 模型化
mod_glm = smf.glm("result ~ hours", data = test_result, 
                  family=sm.families.Binomial()).fit()


# ### 4. 皮尔逊残差

# In[3]:


# 计算皮尔逊残差

# 预测的成功概率
pred = mod_glm.predict()
# 响应变量 (合格情况)
y = test_result.result

# 皮尔逊残差
peason_resid = (y - pred) / sp.sqrt(pred * (1 - pred))
peason_resid.head(3)


# In[4]:


# 获取皮尔逊残差
mod_glm.resid_pearson.head(3)


# In[5]:


# 皮尔逊残差的平方和
sp.sum(mod_glm.resid_pearson**2)


# In[6]:


# 同样出现在 summary 函数的结果中
mod_glm.pearson_chi2


# ### 9. 偏差残差

# In[7]:


# 计算偏差残差

# 预测的成功概率
pred = mod_glm.predict()
# 响应变量 (合格情况)
y = test_result.result

# 与完美预测了合格情况时的对数似然度的差值
resid_tmp = 0 - sp.log(
    sp.stats.binom.pmf(k = y, n = 1, p = pred))
# 偏差残差
deviance_resid = sp.sqrt(
    2 * resid_tmp
) * np.sign(y - pred)
# 打印结果
deviance_resid.head(3)


# In[8]:


mod_glm.resid_deviance.head(3)


# In[9]:


# deviance
sp.sum(mod_glm.resid_deviance ** 2)

