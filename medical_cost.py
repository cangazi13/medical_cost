# -*- coding: utf-8 -*-
"""
Created on Sun Sep 25 18:38:21 2022

@author: sobee
"""

#libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


#%%dataset review
datai=pd.read_csv("../input/insurance/insurance.csv")
datai.head()
datai.info() #7 features
datai.isnull().sum() #not null
datai.describe()


#%%basic data analysis
def bar_plot(variable):

    #get feature
    var = datai[variable]
    #count number of categorical variable
    varvalue = var.value_counts()
    
    #visualize
    plt.figure(figsize=(9,3))
    plt.bar(varvalue.index, varvalue)
    plt.xticks(varvalue.index, varvalue.index.values)
    plt.ylabel("Frequency")
    plt.title(variable)
    plt.show()
    print("{}: \n {}".format(variable,varvalue))
    
category1 = {"sex","age","smoker","children","region"}
for c in category1:
    bar_plot(c)

#%%sex vs charges with smoker
sns.boxplot(x="sex", y="charges",
            hue="smoker", palette=["m", "g"],
            data=datai)
sns.despine(offset=10, trim=True)

#%%smoker counting with sex
sns.catplot(x="smoker", kind="count",hue = 'sex', data=datai)

#%%Treatment costs paid by families with children according to the number of children(male vs female)
childf=datai[datai["children"]>0]
childf0=datai[datai["children"]==0]

sns.boxplot(x=childf["sex"], y=childf["charges"],
            hue=childf["children"], palette="pastel",
            data=datai)
sns.despine(offset=10, trim=True)

#%%Treatment costs paid by families without children(male vs female)
sns.boxplot(x=childf0["sex"], y=childf0["charges"],
            hue=childf0["children"], palette=["b"],
            data=datai)
sns.despine(offset=10, trim=True)

#%%children vs charges
g=sns.factorplot(x="children",y="charges",data=datai,kind="bar",height=6)
g.set_ylabels("Charges")
plt.show()

#%%bmi counting with smoker
f, ax = plt.subplots(figsize=(7, 5))
sns.despine(f)

sns.histplot(
    datai,
    x="bmi", hue="smoker",
    multiple="stack",
    palette="light:m_r",
    edgecolor=".3",
    linewidth=.5,
    log_scale=True,
)
plt.show()

#%%age vs charges with smoker
sns.lmplot(x="age", y="charges", hue="smoker", data=datai, palette = 'rocket')

#%%charges counting with smoker
g=sns.FacetGrid(datai,col="smoker")
g.map(sns.distplot,"charges",bins=25)
plt.show()

#%%smoker vs charges
plt.figure(figsize=(6,6))
sns.boxplot(data=datai, x='smoker', y='charges', palette="husl")
plt.title('Smoker charges')
plt.show()

#%%bmi vs charges with sex
sns.lmplot(x="bmi", y="charges", hue="sex", data=datai, palette = 'dark')

#%%label encoder
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

le.fit(datai["sex"].drop_duplicates()) 
datai["sex"] = le.transform(datai["sex"])

le.fit(datai["smoker"].drop_duplicates()) 
datai["smoker"] = le.transform(datai["smoker"])

#%%correlation
list1=["age","bmi","children","charges","smoker","sex","region"]
sns.heatmap(datai[list1].corr(),annot=True,fmt=".3f")
plt.show()

#%%detect outliers
from collections import Counter
def detect_outliers(df,features):
    outlier_indices=[]
    
    for c in features:
        # 1st quartile
        Q1=np.percentile(df[c],25)
        # 3rd quartile
        Q3=np.percentile(df[c],75)
        # IQR
        IQR=Q3-Q1
        # Outlier step
        outlier_step=IQR-1.5
        # detect outlier and their indeces
        outlier_list_col=df[(df[c]<Q1-outlier_step)|(df[c]>Q3+outlier_step)].index
        # store indeces
        outlier_indices.extend(outlier_list_col)
        
    outlier_indices=Counter(outlier_indices)
    multiple_outliers=list(i for i,v in outlier_indices.items() if v>1)
    return multiple_outliers

datai.loc[detect_outliers(datai,["age","bmi","children"])]

#%%drop outliers
datai=datai.drop(detect_outliers(datai,["age","bmi","children"]),axis=0).reset_index(drop=True)

y=datai["charges"].values
x_data=datai.drop(["charges","region"],axis=1)

#%%normalization
x=(x_data-np.min(x_data))/(np.max(x_data)-np.min(x_data)).values

#%%train_test_split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=10)

#%%linear regression
from sklearn.linear_model import LinearRegression
linr=LinearRegression()
linr.fit(x_train,y_train)
linr_predtr=linr.predict(x_train)
linr_predte=linr.predict(x_test)
print(linr.score(x_test,y_test)) #0.737

#%%bayessian_ridge, decision_tree and random_forest libraries
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import r2_score,mean_squared_error

dt=DecisionTreeRegressor()
rf=RandomForestRegressor()
br=BayesianRidge()

#%%fitting data
dt.fit(x_train,y_train)
rf.fit(x_train,y_train)
br.fit(x_train,y_train)

#%%predict
dt_predtr=dt.predict(x_train)
rf_predtr=rf.predict(x_train)
br_predtr=br.predict(x_train)

dt_pred=dt.predict(x_test)
rf_pred=rf.predict(x_test)
br_pred=br.predict(x_test)

#%%compare results
print("Decision Tree Regressor")
print("")
print('MSE train data: %.3f, MSE test data: %.3f' % (
mean_squared_error(y_train,dt_predtr),
mean_squared_error(y_test,dt_pred))) #MSE train data: 0.000, MSE test data: 38044172.814
print('R2 train data: %.3f, R2 test data: %.3f' % (
r2_score(y_train,dt_predtr),
r2_score(y_test,dt_pred))) #R2 train data: 1.000, R2 test data: 0.747

print("**************************************")

print("Random Forest Regressor")
print("")
print('MSE train data: %.3f, MSE test data: %.3f' % (
mean_squared_error(y_train,rf_predtr),
mean_squared_error(y_test,rf_pred))) #MSE train data: 3497844.585, MSE test data: 24958825.129
print('R2 train data: %.3f, R2 test data: %.3f' % (
r2_score(y_train,rf_predtr),
r2_score(y_test,rf_pred))) #R2 train data: 0.976, R2 test data: 0.834

print("**************************************")

print("Bayessian Ridge")
print("")
print('MSE train data: %.3f, MSE test data: %.3f' % (
mean_squared_error(y_train,br_predtr),
mean_squared_error(y_test,br_pred))) #MSE train data: 35458069.862, MSE test data: 39480860.694
print('R2 train data: %.3f, R2 test data: %.3f' % (
r2_score(y_train,br_predtr),
r2_score(y_test,br_pred))) #R2 train data: 0.754, R2 test data: 0.737
