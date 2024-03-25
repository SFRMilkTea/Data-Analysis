import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import scipy

import numpy as np
from scipy import stats

# Задаем выборку
data = np.array([4.0, 4.6, 4.4, 2.5, 4.8, 5.9, 3.0, 6.1, 5.3, 4.4])
# приемлемая вероятность ошибки
alpha = 0.05
# среднее значение в нулевой гипотезе
null_mean = 5.4
# Вычисляем среднее значение выборки
mean = sum(data) / len(data)
# Вычисляем стандартное отклонение выборки
std_dev = stats.tstd(data)
# Вычисляем t-критерий
t = stats.ttest_1samp(data, popmean=5.4)
# Вычисляем вероятность ошибки
# p = stats.t.sf(abs(t), len(data) - 1) * 2
print(t)

#
#
# # Задаем выборку A
# dataA = np.array(
#     [33.6, 35.3, 36.0, 37.6, 35.9, 33.2, 36.3, 37.1, 37.9, 39.6, 35.1, 36.6, 34.0, 31.2, 39.4,
#      35.4, 32.7, 33.1, 36.0, 37.6, 35.9, 33.2, 36.3, 37.1, 37.9, 39.6, 33.6, 35.3, 36.0, 37.6])
# # Задаем выборку B
# dataB = np.array(
#     [36.6, 36.9, 35.4, 38.0, 37, 37.7, 36.8, 35.1, 37.3, 35.2, 36.3, 36.0, 35.4, 34.7, 36.7,
#      37.0, 36.1, 37.3, 36.6, 36.9, 35.4, 38.0, 37, 37.7, 36.8, 35.1, 37.3, 35.2, 36.8, 35.1])
#
# # Вычисляем t-статистику и p-значение
# t_statistic, p_value = stats.ttest_ind(dataB, dataA)
# # Выводим результаты
# print("t:", t_statistic)
# print("p:", p_value)
# # Визуализируем данные
# plt.figure(figsize=(10, 6))
#
# import statsmodels.api as sm
# import matplotlib.pyplot as plt
#
# # fig = sm.qqplot(dataA, loc=20, scale=5, line='q')
# # fig1 = sm.qqplot(dataB, loc=20, scale=5, line='q')
# # plt.show()
#
# stat, p = scipy.stats.shapiro(dataB) # тест Шапиро-Уилк print(‘Statistics=%.3f, p-value=%.3f’ % (stat, p))
# alpha = 0.05
# print(p)
# if p > alpha:
#  print('Принять гипотезу о нормальности')
# else:
#  print('Отклонить гипотезу о нормальности')
#
#
#
#
# # Выводим гистограммы для каждой выборки
# plt.hist(dataA, alpha=0.5, label='Выборка 1')
# plt.hist(dataB, alpha=0.5, label='Выборка 2')
#
# # Добавляем легенду и название графика
# plt.legend()
# plt.title('Гистограмма выборок')
#
# # Выводим вертикальную линию для среднего значения
# plt.axvline(x=np.mean(dataA), color='blue', linestyle='--', label='Среднее выборки 1')
# plt.axvline(x=np.mean(dataB), color='orange', linestyle='--', label='Среднее выборки 2')
#
# # Отображаем график
# plt.show()
