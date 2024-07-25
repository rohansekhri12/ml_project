# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 10:21:09 2023

@author: Rohan Sekhri
"""
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import explained_variance_score
def feed(data_arr):
    df=pd.read_csv("Data_Pre.csv")
    x=df.iloc[:,0:3].values
    y=df.iloc[:,-1].values
    print("Before",x)
    abnormal(x,y,df,data_arr)
def abnormal(x,y,df,data_arr):
    sim=SimpleImputer(missing_values=np.nan, strategy="most_frequent")
    sim.fit(x[:,0:2])
    x[:,0:2]=sim.transform(x[:,0:2])
    lab=LabelEncoder()
    x[:,2]=lab.fit_transform(x[:,2])
    y[:,]=lab.fit_transform(y[:,])
    print("After:",x)
    algo(x,y,df,data_arr)
def algo(x,y,df,data_arr):
    lr=LinearRegression()
    lr.fit(x,y)
    y_var=lr.predict(x)
    y_pred=lr.predict([data_arr]).round()
    
    print("Prediction: ",y_pred)
    if (y_pred==1):
        print(" Patient is diabetic")
    else:
        print("Patient is  not diabetic")
    print("Analysis!!")
    print(lr.score(x,y))
    expp=explained_variance_score(y,y_var)
    print("Algo_Score: ",expp)    

def main():
    print("PLEASE ENTER THE PATIENT DETAILS !!")
   
    age=int(input("Enter Patient age = "))
    sysb=int(input("Enter Patient's Sysblood pressure= "))
    print('''Enter Patient's Pulse Rate?
          0.for High 
          1.for Low
          2.for Medium''')
    pul=int(input("Enter your choice:0/1/2 = "))
    data_arr=[age,sysb,pul]
    feed(data_arr)
    
main()    