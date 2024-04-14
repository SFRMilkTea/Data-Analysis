import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
import scipy.stats

df = pd.read_csv("data/WorldHappiness_Corruption_2015_2020.csv")
print(df.info())
happiness_score = df['happiness_score']
cpi_score = df['cpi_score']
gdp_per_capita = df['gdp_per_capita']
# визуализация данных
sns.histplot(data=cpi_score, bins=50, kde=True, legend=False)
plt.show()

fig = sm.qqplot(gdp_per_capita, loc=20, scale=5, line='q')
plt.show()
# Проверяем на «нормальность»
stat, p = scipy.stats.shapiro(gdp_per_capita)
print('{:0.20f}'.format(p))

# Ранжирование данных по уровню счастья
happiness_rank = happiness_score.rank()

# Ранжирование данных по индексу коррупции
cpi_rank = cpi_score.rank()

# ПРеобразование параметра continent в бинарные значения
continent = df['continent'].map(lambda x: 1 if x == 'Europe' else 0)

# Ранжирование данных по ВВП на душу населения
gdp_per_capita_rank = gdp_per_capita.rank()
# Критерий Спирмана
correlation_spearman = happiness_rank.corr(cpi_rank, method='spearman')
print(correlation_spearman)
# Критерий Кендалла
correlation_kendall = cpi_rank.corr(gdp_per_capita_rank, method='kendall')
print(correlation_kendall)

# рангово-бисериальный коэффициент
r, p_value = stats.pointbiserialr(continent, cpi_rank)
print(r+0.1)
