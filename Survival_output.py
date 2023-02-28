import pandas as pd
from tabulate import tabulate
import numpy as np
from datetime import datetime
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
import scipy.stats as st
import scipy

import platform

def mean_confidence_interval(data, confidence=0.95):
    data = data[[ str(k+1)+'-fold' for k in range(10)]].values
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m-h, m+h

dataLocation = "C:\\Users\\hrmor\\OneDrive - University of Mississippi Medical Center\\03_UMMC\\Projects__Jackson heart\\JHS data (Moradi-Morris)\\"
resultsLocation = ""
if platform.system() != 'Windows':
    dataLocation = "/home/hmoradi/Downloads/Data/JHS/"
    resultsLocation = "/home/hmoradi/Downloads/PycharmProject/JacksonHeart/"


results = pd.read_csv("Result_10fold_main_FULL" + ".csv")
print(results.head(2))

results['95_Percentile_L'] = results.apply(lambda row : mean_confidence_interval(row)[0], axis=1)
results['95_Percentile_H'] = results.apply(lambda row : mean_confidence_interval(row)[1], axis=1)
print(results.head(2))


results.to_csv(resultsLocation +  "Result_Final" + ".csv", index=True)


