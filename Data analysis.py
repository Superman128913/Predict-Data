

import pandas
from sklearn import linear_model
import numpy as np

import math
import matplotlib.pyplot as plt

df = pandas.read_csv("predicted data.csv")

df_year = df.groupby('settlement_date').sum()


df_settlement_period_bsous = df.groupby('settlement_period').max()
df_settlement_period_bsous = df_settlement_period_bsous['bsuos']



df_big_bsuos_count = df.groupby('settlement_period')['bsuos'].apply(lambda x: (x>6).sum())/df.groupby('settlement_period')['bsuos'].count()
plt.title("Ratio of events (value of bsuos > 6) in each settlement period!")
plt.plot(df_big_bsuos_count)
plt.savefig('Ratio of events1.png')
plt.show()

bsuos_year = df_year['bsuos']


plt.title("Sum of bsuos in each day")
bsuos_year = bsuos_year[1:8]
plt.plot(bsuos_year)
plt.savefig("Sum of bsuos in each day1.png")
plt.show()


df_demand = df['dadf_mw_value']
plt.title('Variation of demand for electricity ')
plt.plot(df_demand)
plt.savefig('Variation of demand for electricity1.png')
plt.show()



df_other = df['residual_mw_value']
plt.title('Variation of other resources')
plt.plot(df_other)
plt.savefig('Variation of other resources1.png')
plt.show()


