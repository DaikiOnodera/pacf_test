#!/usr/bin/env python
# encoding:utf-8

import numpy as np
import pandas as pd
from scipy import stats

from matplotlib import pylab as plt
import seaborn as sns
sns.set()

from matplotlib.pylab import rcParams
rcParams["figure.figsize"] = 15, 6
import statsmodels.api as sm

dataNormal = pd.read_csv("AirPassengers.csv")
dateparse = lambda dates:pd.datetime.strptime(dates, "%Y-%m")
data = pd.read_csv("AirPassengers.csv", index_col="Month", \
    date_parser=dateparse, dtype="float")

ts = data["#Passengers"]

ts_acf = sm.tsa.stattools.acf(ts, nlags=40)
ts_pacf = sm.tsa.stattools.pacf(ts, nlags=40, method="ols")
print(ts_pacf)
