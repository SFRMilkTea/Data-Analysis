import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
import scipy.stats
from sklearn import preprocessing
from scipy import stats

df = pd.read_excel("data/lb5.xlsx")

from statsmodels.formula.api import ols

model = ols('Y ~ X1', data=df).fit()
# print(model.summary())
# fig = plt.figure(figsize=(12,8))
# fig = sm.graphics.plot_regress_exog(model, 'X1', fig=fig)
# plt.show()
#
# #define residuals
res = model.resid

# #create Q-Q plot
# fig = sm.qqplot(res, loc = 20, scale = 5 ,  line='q')
# plt.show()

# Check for "normality" / Проверяем на «нормальность»
stat, p = stats.shapiro(res) # тест Шапиро-Уилка
alpha = 0.05
print(p)
if p > alpha:
    print('Принять гипотезу о нормальности')
else:
    print('Отклонить гипотезу о нормальности')