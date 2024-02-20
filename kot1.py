import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
from time import sleep

df = pd.read_excel("data/data.xlsx", sheet_name="Data")
sns.histplot(data=df['наблюдения'], bins=50, kde=True, legend=False)
plt.show()
sns.boxplot(data=df)
plt.show()
data_mean = np.mean(df)
print(f'Среднее выборочное: {round(data_mean, 2)}')

data_mode = str(stats.mode(df).mode[0])
print(f'Мода: {data_mode}')

data_median = np.median(df)
print(f'Медиана: {int(data_median)}')

data_std = np.std(df['наблюдения'], axis=0)
print(f'Стандартное отклонение: {round(data_std, 2)}')

data_skew = stats.skew(df, bias=False)
print(f'Коэффициент асимметрии: {round(data_skew[0], 2)}')

data_kurt = stats.kurtosis(df, bias=False)
print(f'Коэффициент эксцесса: {round(data_kurt[0], 2)}')

import statsmodels.api as sm

fig = sm.qqplot(df['наблюдения'], loc=20, scale=5, line='q')
plt.show()
# Проверяем на «нормальность»
stat, p = stats.shapiro(df['наблюдения'])
alpha = 0.05
print(round(p, 4))
print(stats.t.interval(1 - alpha, len(df['наблюдения']) - 1, loc=data_mean, scale=stats.sem(df['наблюдения'])))
import scipy.stats

t = scipy.stats.t.ppf(1-.05/2, len(df['наблюдения']) - 1)
n = round(((t * np.std(df['наблюдения'], ddof=1))/ 0.5) ** 2)
print(f'n: {n}')
CV = (data_std / data_mean) * 100
print(CV)