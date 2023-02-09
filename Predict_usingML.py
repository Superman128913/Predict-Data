

import pandas
from sklearn import linear_model
import numpy as np

import math
import matplotlib.pyplot as plt

df = pandas.read_csv("bsuos_data.csv")

X = df[['residual_mw_value', 'bsuos']]
y = df[['wind_mw_value','dadf_mw_value']]

print("preparing training data")

new_train_data_x  = []
new_tarain_data_y = []
for i in range(len(df)):
    if (not math.isnan(df.iat[i,2])) and (not math.isnan(df.iat[i,3])):
        new_train_data_x.append(i)
        new_tarain_data_y.append( [df.iat[i,2],df.iat[i,3]])
        

new_train_data_x = np.array(new_train_data_x)
new_x = new_train_data_x.reshape(-1,1)
new_tarain_data_y = np.array(new_tarain_data_y)
print(new_x)
print(new_tarain_data_y)
print("traning model... ...")

regr = linear_model.LinearRegression()
regr.fit(new_x, new_tarain_data_y)
print("good, model trained")
print("------------------------------")
for i in range(len(df)):
    if math.isnan(df.iat[i,2]):
        df.iat[i,2] = regr.predict([[i]])[0,0]
    if math.isnan(df.iat[i,3]):
        df.iat[i,3] = regr.predict([[i]])[0,1]

df.to_csv("predicted data.csv")


        
print("New data was successfully saved!!!")


'''
winmv = input("please insert win_mv")
dadfmv = input("please insert dedf_mv")
residualmv = input("please insert residual_mv")   

predict_bsuos = regr.predict([[winmv, dadfmv,residualmv]])
print("pridicted value of bsuos: ", predict_bsuos)
#predict the CO2 emission of a car where the weight is 2300kg, and the volume is 1300cm3:
'''
