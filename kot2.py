import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# загружаем датасет
df = sns.load_dataset('iris')
print(f"количество строк: {df.shape[0]}, количество признаков: {df.shape[1]}")
print("Первые пять строк датасета:")
print(df.head())
# информация о датасете
print(df.info())
# количество ирисов каждой разновидности
print(df.groupby('species').size())

# построим круговую диаграмму
colors = sns.color_palette('bright')
# подписи для диаграммы
data_names = df['species'].unique()
df['species'].value_counts().plot.pie(
    explode=[0.05, 0.05, 0.05], autopct='%1.1f%%', colors=colors, shadow=True,
    figsize=(6, 6), labels=None)
plt.title('Круговая диаграмма по видам ирисов')
# легенда
plt.legend(
    bbox_to_anchor=(-0.16, 0.45, 0.25, 0.25),
    loc='best', labels=data_names)
plt.show()
sns.countplot(x='species', data=df)
plt.show()
sns.countplot(x='species', data=df, hue='sepal_length')
plt.show()
import pandas as pd
bins = [4, 5, 6, 7, 8]
grouped = df.groupby([pd.cut(df['sepal_length'], bins), 'species']).size().unstack(fill_value=0)
grouped.plot(kind='bar')
plt.show()

plt.figure(figsize=(12, 9))
plt.subplot(2, 2, 1)
sns.barplot(x='species', y='petal_length', data=df, palette="Set1")

plt.subplot(2, 2, 2)
sns.barplot(x='species', y='petal_width', data=df, palette="Set1")

plt.subplot(2, 2, 3)
sns.barplot(x='species', y='sepal_length', data=df, palette="Set1")

plt.subplot(2, 2, 4)
sns.barplot(x='species', y='sepal_width', data=df, palette="Set1")
plt.show()
# Построим распределение параметров ирисов
# Установим стиль для графиок библиотеки seaborn
sns.set(style='whitegrid')
# Создание объектов фигуры и оси
fig, axs = plt.subplots(2, 2, figsize=(15, 10))
# Построим распределение для параметров пингвинов.
sns.histplot(data=df, x='petal_length', kde=True, color='blue', hue='species', ax=axs[0,0])
sns.histplot(data=df, x='petal_width', kde=True, color='green', hue='species', ax=axs[0,1])
sns.histplot(data=df, x='sepal_length', kde=True, color='red', hue='species', ax=axs[1,0])
sns.histplot(data=df, x='sepal_width', kde=True, color='orange', hue='species', ax=axs[1,1])
# Установим названия для графиков
axs[0,0].set_title('Распределение по длине лепестков')
axs[0,1].set_title('Распределение по ширине лепестков')
axs[1,0].set_title('Распределение по длине чашелистиков')
axs[1,1].set_title('Распределение по ширине чашелистиков')
# Установим общий заголовок
fig.suptitle('Анализ распределения параметров ирисов')
# Установим промежутки между графиками
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.4, hspace=0.4)
# Нарисуем графики
plt.show()
#
# Создание объектов фигуры и оси
fig, axs = plt.subplots(2, 2, figsize=(15, 10))
# Построим распределение для параметров пингвинов.
sns.stripplot(data=df, x='species', y='petal_length',  size=8, edgecolor="gray", palette="Set1", linewidth=2, ax=axs[0,0])
sns.stripplot(data=df, x='species', y='petal_width', size=8, edgecolor="gray", palette="Set1", linewidth=2, ax=axs[0,1])
sns.swarmplot(data=df, x='species', y='sepal_length', size=8, edgecolor="gray", palette="Set1", linewidth=2, ax=axs[1,0])
sns.swarmplot(data=df, x='species', y='sepal_width',  size=8, edgecolor="gray", palette="Set1", linewidth=2, ax=axs[1,1])
# Установим названия для графиков
axs[0,0].set_title('Распределение по длине лепестков')
axs[0,1].set_title('Распределение по ширине лепестков')
axs[1,0].set_title('Распределение по длине чашелистиков')
axs[1,1].set_title('Распределение по ширине чашелистиков')
# Установим общий заголовок
fig.suptitle('Анализ параметров ирисов')
# Установим промежутки между графиками
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.4, hspace=0.4)
# Нарисуем графики
plt.show()
# ящик с усами
# Установим стиль для графиок библиотеки seaborn
sns.set(style='darkgrid')
# Создание объектов фигуры и оси
fig, axs = plt.subplots(2, 2, figsize=(15, 10))
# Установим палитру
sns.set_palette('Set2')
# Фильтруем DataFrame только для вида 'virginica'
df_virginica = df[df['species'] == 'virginica']
# Построим распределения
sns.boxplot(data=df_virginica, x='species', y='petal_length', ax=axs[0,0])
sns.boxplot(data=df_virginica, x='species', y='petal_width', ax=axs[0,1])
sns.boxplot(data=df_virginica, x='species', y='sepal_length', ax=axs[1,0])
sns.boxplot(data=df_virginica, x='species', y='sepal_width', ax=axs[1,1])

# Установим названия для графиков
axs[0,0].set_title('Распределение по длине лепестков')
axs[0,1].set_title('Распределение по ширине лепестков')
axs[1,0].set_title('Распределение по длине чашелистиков')
axs[1,1].set_title('Распределение по ширине чашелистиков')
# Установим общий заголовок
fig.suptitle('Анализ параметров ирисов вида virginica')
# Установим промежутки между графиками
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.4, hspace=0.4)
# Нарисуем графики
plt.show()

# Зададим свой набор цветов:
cmap={'setosa':'y', 'versicolor':'g', 'virginica': 'r'}
# построим диаграмму рассеяния
sns.scatterplot(x='petal_length', y='petal_width', hue='species', style='species', palette=cmap, data=df)
plt.show()
sns.set_style(style='white')
sns.pairplot(data=df,hue='species',palette=['#6baddf','#98FB98','#9932CC'])
plt.show()
sns.set_style(style='white')
sns.pairplot(data=df, hue='species', kind='reg', diag_kind='hist', vars=["sepal_length", "petal_length"], corner=True,
             palette=['#6baddf', '#98FB98', '#9932CC'])
plt.show()
sns.set_palette('rainbow')
sns.set_style("whitegrid")
sns.regplot(x="petal_length", y="petal_width", data=df)
plt.show()
sns.lmplot(x="petal_length", y="petal_width", hue="species", palette="Set1", data=df)
plt.show()
sns.lmplot(x="petal_length", y="petal_width", palette="Set1", hue="species", col="species",
           facet_kws={'sharex':False, 'sharey':False}, data=df)
plt.show()
#
sns.jointplot(x="petal_length", y="petal_width", hue="species", height=5, data=df)
plt.show()

sns.jointplot(x="petal_length", y="petal_width", hue="species", height=5, data=df, kind='hist')
plt.show()

g = sns.jointplot(data=df, x="petal_length", y="petal_width")
g.plot_joint(sns.kdeplot, color="r", zorder=0, levels=6)
g.plot_marginals(sns.rugplot, color="r", height=-.15, clip_on=False)
plt.show()
df.corr(method='pearson', numeric_only=True)
print(df.corr(method='pearson', numeric_only=True))
sns.heatmap(df.corr(method='pearson', numeric_only=True), annot=True)
plt.show()
