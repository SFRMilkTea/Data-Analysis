import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
import scipy.stats
from sklearn import preprocessing
from scipy import stats

df = pd.read_csv("data/WorldHappiness_Corruption_2015_2020.csv")

happiness_score = df['happiness_score']
cpi_score = df['cpi_score']

# датафрейм
data = pd.DataFrame({'happiness_score': happiness_score, 'cpi_score': cpi_score})
#
# data_scatter = sns.scatterplot(x='cpi_score', y='happiness_score', data=df)
# data_scatter.set(title='Диаграмма рассеяния', xlabel='Индекс восприятия коррупции', ylabel='Уровень счастья')
# plt.show()
#
# stat, p = stats.shapiro(happiness_score)
# alpha = 0.05
# print(p)
# if p > alpha:
#     print('Принять гипотезу о нормальности для уровня счастья')
# else:
#     print('Отклонить гипотезу о нормальности для уровня счастья')
# stat, p = stats.shapiro(cpi_score)
# print(p)
# if p > alpha:
#     print('Принять гипотезу о нормальности для индекса восприятия коррупции')
# else:
#     print('Отклонить гипотезу о нормальности для индекса восприятия коррупции')

#
# # Ранжирование данных по уровню счастья
# happiness_rank = happiness_score.rank()
# # Ранжирование данных по индексу коррупции
# cpi_rank = cpi_score.rank()
# # Коэффициент Спирмена
# correlation_spearman = happiness_rank.corr(cpi_rank, method='spearman')
# print(correlation_spearman)


# # Вычисляем стандартные отклонения
# SD_x = np.std(happiness_score)
# SD_y = np.std(cpi_score)

# # Вычисляем наклон прямой
# b1 = r * (SD_y/ SD_x)
# print(f"Угол наклона прямой: {b1}")
# # Вычисляем пересечение с осью y
# b0 = np.mean(cpi_score - b1 * np.mean(happiness_score))
# print(f"Отрезок прямой: {b0}")
#
# import scipy.stats as sps
#
# reg = sps.linregress(happiness_score, cpi_score)
# print(f'Scipy function: a = {round(reg.intercept, 3)}; b = {round(reg.slope, 3)}')
#
# cpi_score_predict = [(reg.intercept + reg.slope * x) for x in cpi_score]
# line = f'Regression line: y={reg.intercept:.2f}+{reg.slope:.2f}x'
# fig, ax = plt.subplots()
# ax.plot(cpi_score, happiness_score, linewidth=0, marker='o', label='Data points')
# ax.plot(cpi_score, cpi_score_predict, label=line)
# ax.set_xlabel('Индекс восприятия коррупции')
# ax.set_ylabel('Уровень счастья')
# ax.legend(facecolor='white')
# plt.show()

# import scipy.stats as sps
#
# reg = sps.linregress(cpi_score, happiness_score)
# print(f'Scipy function: a = {round(reg.intercept, 3)}; b = {round(reg.slope, 3)}')
#
# cpi_score_predict = [(reg.intercept + reg.slope * x) for x in cpi_score]
# line = f'Regression line: y={reg.intercept:.2f}+{reg.slope:.2f}x'
# fig, ax = plt.subplots()
# ax.plot(cpi_score, happiness_score, linewidth=0, marker='o', label='Data points')
# ax.plot(cpi_score, cpi_score_predict, label=line)
# ax.set_xlabel('cpi_score')
# ax.set_ylabel('happiness_score')
# ax.legend(facecolor='white')
# plt.show()

from statsmodels.formula.api import ols

model = ols('cpi_score ~ happiness_score', data=df).fit()
# print(model.summary())
#
#

# from statsmodels.formula.api import ols
#
# model = ols('cpi_score ~ happiness_score', data=df).fit()
# print(model.summary())
#
# fig = plt.figure(figsize=(12,8))
# fig = sm.graphics.plot_regress_exog(model, 'happiness_score', fig=fig)
# plt.show()

# Добавляем константу для уравнения регрессии
X = happiness_score
y = cpi_score
# X = sm.add_constant(X)

# # Вычисление параметров регрессии
model = sm.OLS(y, X).fit()
# intercept, slope = model.params
#
# # Вычисление доверительных интервалов для параметров
# conf_int = model.conf_int()
# conf_int_intercept = conf_int.loc['const']
# conf_int_slope = conf_int.loc['happiness_score']
#
# print("Коэффициенты регрессии:")
# print("Intercept (константа):", intercept)
# print("Slope (наклон):", slope)
# print("\nДоверительные интервалы:")
# print("Доверительный интервал для константы:", conf_int_intercept.values)
# print("Доверительный интервал для наклона:", conf_int_slope.values)

#define residuals
res = model.resid
#
# #create Q-Q plot
# fig = sm.qqplot(res, loc = 20, scale = 5 ,  line='q')
# plt.show()

# Check for "normality" / Проверяем на «нормальность»
# stat, p = stats.shapiro(res) # тест Шапиро-Уилка
# alpha = 0.05
# print(p)
# if p > alpha:
#  print('Accept the normality hypothesis / Принять гипотезу о нормальности')
# else:
#  print('Reject the normality hypothesis /Отклонить гипотезу о нормальности')

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = X_train.values.reshape(-1, 1)
X_test = X_test.values.reshape(-1, 1)


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

model = LinearRegression()
model.fit(X_train,y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Среднеквадратичная ошибка на тестовом наборе: {mse}')

from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X_test, y_test, cv=5, scoring='neg_mean_squared_error')
average_mse = -scores.mean()
print(f'Среднеквадратичная ошибка при использовании 5-кратной кросс-валидации: {average_mse}')

