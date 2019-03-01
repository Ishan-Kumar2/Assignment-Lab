# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 10:58:04 2019

@author: Ishan Kumar
"""

import numpy as np
import matplotlib.pyplot as pt
import pandas as pd

from sklearn.datasets import load_boston
boston = load_boston()

print(boston.keys())

bos = pd.DataFrame(boston.data)
print(bos.head())
bos.columns = boston.feature_names

bos['medval'] = boston.target
print(bos.head())


#FITTING LINEAR REGRESSION MODEL
X1=bos['LSTAT']
X1=bos.LSTAT.values.reshape(-1,1)
Y=bos.medval.values
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X1,Y)

#PLOT DATA USING SUBPLOT
fig, ax = pt.subplots(figsize=(8,6))

ax.scatter(X1, Y,facecolors='none', edgecolors='b',label="data");
ax.set_ylabel('MEDVALUE');
ax.set_xlabel('LSTAT');
ax.plot(X1,regressor.predict(X1),color='red')
ax.show()

#CREATING SUMMARY TABLE OF THE LINEAR MODEL 
import statsmodels.api as sm
X=sm.add_constant(X)
regressor=sm.OLS(Y,X)
regressorfit=regressor.fit()
print(regressorfit.summary())
#Getting confidence intervals
from statsmodels.sandbox.regression.predstd import wls_prediction_std
prstd, iv_l, iv_u = wls_prediction_std(regressorfit)
from statsmodels.stats.outliers_influence import summary_table
simpleTable, data, column_names = summary_table(regressorfit, alpha=0.05)

#Plotting the confidence intervals over the MAIN plot
predicted_mean_ci_low, predicted_mean_ci_high = data[:,4:6].T
fig, ax = pt.subplots(figsize=(8,6))
ax.plot(x, iv_u, color='0.75',label="Prediction Interval")
ax.plot(x, iv_l, color='0.75')

ax.plot(x,predicted_mean_ci_low, 'r', label="Predicted Mean CI")
ax.plot(x,predicted_mean_ci_high,'r')


ax.scatter(X1, Y,facecolors='none', edgecolors='b',label="data");
ax.set_ylabel('MEDVALUE');
ax.set_xlabel('LSTAT');
ax.plot(X1,regressor.predict(X1),color='red')
ax.show()



#Multiple Regression
X2=bos.iloc[:,:-1].values
from sklearn.linear_model import LinearRegression
Regressor=LinearRegression()
Regressor.fit(X2,Y)

#Summary for Multiple Regression
import statsmodels.api as sm
X=sm.add_constant(X)
regressor=sm.OLS(Y,X2)
regressorfit=regressor.fit()
print(regressorfit.summary())


#Including interaction terms like ab
import statsmodels.formula.api as smf

model = smf.ols(formula='medval ~ LSTAT*AGE', data=bos)
estimate = model.fit()
#Summary with interaction terms
print(estimate.summary())


#POLYNOMIAL REGRESSION WITH DEGREE 2
from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=2)
X_poly=poly_reg.fit_transform(X1)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_poly,Y)

#Plotting the polynomial Regression
fig, ax = pt.subplots(figsize=(8,6))

ax.scatter(X1, Y,facecolors='none', edgecolors='b',label="data");
ax.set_ylabel('MEDVALUE');
ax.set_xlabel('LSTAT');
ax.plot(X1,regressor.predict(X_poly),color='red')
ax.show()





