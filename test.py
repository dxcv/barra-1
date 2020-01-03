
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

#print(pd.Timestamp(datetime.today()).strftime('%Y-%m-%d'))

t = pd.DataFrame()
t['a'] = [1,2,6]
t['b'] = [3,4,5]
print(t)
print(t.std()['a'])
#t = np.matrix([[1,2,3],[1,2,3],[2,3,4],[3,6,7]])
t = np.matrix(t)
print(np.shape(t))
print(np.cov(t.T))
