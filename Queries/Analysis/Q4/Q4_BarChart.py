import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

csvFile = pd.read_csv('./Query/Q4/Q4.csv', sep=',', names=['RepositoryName','CommitId', 'NumOfModifiedFiles'],
                      skiprows=1)

dfCsv = pd.DataFrame(csvFile)

modifiedFilesDf = dfCsv["NumOfModifiedFiles"].dropna().to_numpy()

barChartData = pd.cut(modifiedFilesDf, [0,4,8,11,16,26,50,np.inf], labels=['0-3','4-7','8-10','11-15','16-25','26-49','50+'])
barChart = barChartData.value_counts().plot.bar(color="green")

plt.xticks(rotation=0)
plt.title("Distribution of Modified Files by Commit")
plt.xlabel('Number Of Modified Files')
plt.ylabel('Number Of Commit') 

barChart.bar_label(barChart.containers[0])

plt.show()


