#!/usr/bin/env python
# coding: utf-8

# # 第 4 节　线性模型与神经网络
# 
# ## 第 7 章　统计学与机器学习｜用 Python 动手学统计学
# 
# 

# ### 环境准备

# In[1]:


# 用于数值计算的库
import numpy as np
import pandas as pd
import scipy as sp

# 用于估计统计模型的库 (部分版本会有警告信息)
import statsmodels.formula.api as smf
import statsmodels.api as sm

# 用于多层感知器的库
from sklearn.neural_network import MLPClassifier

# 导入示例数据
from sklearn.datasets import load_iris

# 区分训练集与测试集
from sklearn.model_selection import train_test_split

# 标准化数据
from sklearn.preprocessing import StandardScaler

# 设置浮点数打印精度
get_ipython().run_line_magic('precision', '3')


# ### 读入数据并整形

# In[2]:


# 导入示例数据
iris = load_iris()


# In[3]:


# 解释变量的名称
iris.feature_names


# In[4]:


# 响应变量的名称
iris.target_names


# In[5]:


# 解释变量仅为萼片 (sepal)
X = iris.data[50:150, 0:2]
# 只取2种鸢尾花
y = iris.target[50:150]

print("解释变量行数与列数：", X.shape)
print("响应变量行数与列数：", y.shape)


# In[6]:


# 把数据分为训练集与测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state = 2)

print("解释变量行数与列数：", X_train.shape)
print("响应变量行数与列数：", y_train.shape)


# ### 实现：logistic 回归

# In[7]:


# 打印响应变量
y_train[0:10]


# In[8]:


# 数据整形
# 解释变量的数据帧
X_train_df = pd.DataFrame(
    X_train, columns = ["sepal_len", "sepal_wid"])
# 响应变量的数据帧
y_train_df = pd.DataFrame({"species": y_train - 1})
# 连接数据帧
iris_train_df = pd.concat(
    [y_train_df, X_train_df], axis=1)
# 打印结果
print(iris_train_df.head(3))


# In[9]:


# 模型化
# 长度与宽度模型
logi_mod_full = smf.glm(
    "species ~ sepal_len + sepal_wid", data = iris_train_df,
    family=sm.families.Binomial()).fit()

# 长度模型
logi_mod_len = smf.glm(
    "species ~ sepal_len", data = iris_train_df,
    family=sm.families.Binomial()).fit()

# 宽度模型
logi_mod_wid = smf.glm(
    "species ~ sepal_wid", data = iris_train_df,
    family=sm.families.Binomial()).fit()

# 空模型
logi_mod_null = smf.glm(
    "species ~ 1", data = iris_train_df,
    family=sm.families.Binomial()).fit()

# 对比 AIC
print("full", logi_mod_full.aic.round(3))
print("len ", logi_mod_len.aic.round(3))
print("wid ", logi_mod_wid.aic.round(3))
print("null", logi_mod_null.aic.round(3))


# In[10]:


# 查看估计的系数等指标
logi_mod_len.summary().tables[1]


# In[11]:


# 预测精度
# 数据整形
X_test_df = pd.DataFrame(
    X_test, columns = ["sepal_len", "sepal_wid"])

# 拟合与预测
logi_fit = logi_mod_len.fittedvalues.round(0)
logi_pred = logi_mod_len.predict(X_test_df).round(0)

# 正确数
true_train = sp.sum(logi_fit == (y_train - 1))
true_test = sp.sum(logi_pred == (y_test - 1))

# 命中率
result_train = true_train / len(y_train)
result_test = true_test / len(y_test)

# 打印结果
print("训练集的命中率：", result_train)
print("测试集的命中率：", result_test)


# ### 实现：标准化

# In[12]:


# 准备标准化
scaler = StandardScaler()
scaler.fit(X_train)
# 标准化
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[13]:


sp.std(X_train_scaled, axis=0)


# In[14]:


sp.std(X_test_scaled, axis=0)


# ### 实现：神经网络

# In[15]:


nnet = MLPClassifier(
    hidden_layer_sizes = (100,100),
    alpha = 0.07,
    max_iter = 10000,
    random_state = 0)
nnet.fit(X_train_scaled, y_train)

# 正确数
print("训练集的命中率：", nnet.score(X_train_scaled, y_train))
print("测试集的命中率：", nnet.score(X_test_scaled, y_test))


# In[ ]:




