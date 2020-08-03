# Ramarea, Tumisang
# Summer 2020
# Covid-19: Mitigating Potential Propagation by Truck Drivers in Botswana
# Script to Estimate the value of statistical life for Botswana
# Reference Credit: Adarsh Menon (https://towardsdatascience.com/linear-regression-in-6-lines-of-python-5e1d0cd05b8d) 

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

df = pd.DataFrame(np.array([['Canada',7600000.00,46232.99], ['China',2500000.00,9770.85],
                            ['Italy',6400000.00,34483.20], ['Spain',6100000.00,30370.89],
                           ['UK',7100000.00, 42943.90], ['US', 10000000.00,62794.59]]), 
                  columns = ['Country', 'Viscusi VSL Estimate ($)', '2018 GDP Per Capita ($)'])
GDPPC = df.iloc[:, 2].values.reshape(-1, 1)
VVSL = df.iloc[:, 1].values.reshape(-1, 1)  
lm = LinearRegression()  
lm.fit(GDPPC, VVSL) 
BW_VVSL = linear_regressor.predict(8258.64)  
BW_VVSL
