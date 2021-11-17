#!/usr/bin/env python
# coding: utf-8

# # 第 3 节　含有多个解释变量的模型
# 
# ## 第 5 章　正态线性模型｜用 Python 动手学统计学
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
# 在Jupyter Notebook里显示图形
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# 读入数据
sales = pd.read_csv("5-3-1-lm-model.csv")
print(sales.head(3))


# ### 2. 实现：数据可视化

# In[3]:


# 数据可视化
sns.pairplot(data = sales, hue = "weather", 
             palette="gray")


# ### 3. 错误的分析：建立只有 1 个变量的模型

# In[4]:


# 只使用价格这 1 种解释变量进行建模
lm_dame = smf.ols("sales ~ price", sales).fit()
lm_dame.params


# In[5]:


# 价格的系数与 0 存在显著性差异
print(sm.stats.anova_lm(lm_dame, typ=2))


# In[6]:


# 价格与销售额的关系
sns.lmplot(x = "price", y = "sales", data = sales,
           scatter_kws = {"color": "black"},
           line_kws    = {"color": "black"})


# ### 4. 分析解释变量之间的关系

# In[7]:


# 计算各天气下的均值
print(sales.groupby("weather").mean())


# In[8]:


# 不同天气中销售额—价格回归直线
sns.lmplot(x = "price", y = "sales", data = sales, 
           hue="weather", palette='gray')


# ### 5. 实现：多解释变量的模型

# In[9]:


# 估计多解释变量的模型
lm_sales = smf.ols(
    "sales ~ weather + humidity + temperature + price",
    data=sales).fit()
# 估计的结果
lm_sales.params


# ### 6. 错误的分析：使用普通方差分析

# In[10]:


# 普通方差分析
print(sm.stats.anova_lm(lm_sales, typ=1).round(3))


# In[11]:


# 改变解释变量的顺序
lm_sales_2 = smf.ols(
    "sales ~ weather + temperature + humidity + price",
    data=sales).fit()
# 检验结果
print(sm.stats.anova_lm(lm_sales_2, typ=1).round(3))


# ### 7. 实现：回归系数的 t 检验

# In[12]:


# 模型 1 的回归系数的 t 检验
lm_sales.summary().tables[1]


# In[13]:


# 模型 2 的回归系数的 t 检验
lm_sales_2.summary().tables[1]


# ### 9. 模型选择与方差分析

# In[14]:


# 空模型的残差平方和
mod_null = smf.ols("sales ~ 1", sales).fit()
resid_sq_null = sp.sum(mod_null.resid ** 2)
resid_sq_null


# In[15]:


# 天气模型的残差平方和
mod_1 = smf.ols("sales ~ weather", sales).fit()
resid_sq_1 = sp.sum(mod_1.resid ** 2)
resid_sq_1


# In[16]:


# 残差平方和的差
resid_sq_null - resid_sq_1


# In[17]:


print(sm.stats.anova_lm(mod_1).round(3))


# In[18]:


# "天气 + 湿度" 模型的残差平方和
mod_2 = smf.ols(
    "sales ~ weather + humidity", sales).fit()
resid_sq_2 = sp.sum(mod_2.resid ** 2)
resid_sq_2


# In[19]:


# 残差平方和的差
resid_sq_1 - resid_sq_2


# In[20]:


print(sm.stats.anova_lm(mod_2).round(3))


# In[21]:


# "天气 + 气温" 模型的残差平方和
mod_2_2 = smf.ols(
    "sales ~ weather + temperature", sales).fit()
resid_sq_2_2 = sp.sum(mod_2_2.resid ** 2)
resid_sq_2_2


# In[22]:


# "天气 + 气温 + 湿度" 模型的残差平方和
mod_3_2 = smf.ols(
    "sales ~ weather + temperature + humidity",
    sales).fit()
resid_sq_3_2 = sp.sum(mod_3_2.resid ** 2)
resid_sq_3_2


# In[23]:


resid_sq_2_2 - resid_sq_3_2


# In[24]:


print(sm.stats.anova_lm(mod_3_2).round(3))


# ### 11. 实现：Type II ANOVA

# In[25]:


# 包含所有解释变量的模型的残差平方和
mod_full = smf.ols(
    "sales ~ weather + humidity + temperature + price",
    sales).fit()
resid_sq_full = sp.sum(mod_full.resid ** 2)
resid_sq_full


# In[26]:


# 不含湿度的模型的残差平方和
mod_non_humi = smf.ols(
    "sales ~ weather + temperature + price", 
    sales).fit()
resid_sq_non_humi = sp.sum(mod_non_humi.resid ** 2)
resid_sq_non_humi


# In[27]:


# 调整平方和
resid_sq_non_humi - resid_sq_full


# In[28]:


# Type II ANOVA
print(sm.stats.anova_lm(mod_full, typ=2).round(3))


# In[29]:


# 对比这两个模型
mod_full.compare_f_test(mod_non_humi)


# ### 13. 实现：变量选择与模型选择

# In[30]:


print(sm.stats.anova_lm(mod_non_humi, typ=2).round(3))


# In[31]:


mod_non_humi.params


# ### 14. 实现：用 AIC 进行变量选择

# In[32]:


print("包含所有变量的模型：", mod_full.aic.round(3))
print("不含湿度的模型　　：", mod_non_humi.aic.round(3))


# In[ ]:




