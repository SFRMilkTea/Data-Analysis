import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_excel("data/data.xlsx", header=0)
df.head()
sns.histplot(data=df, kde=True, legend=False)
plt.title('Viscositie/ Вязкость')
plt.xlabel('Time sec/ Время, с')
plt.ylabel('Frequency/ Частота')
plt.show()