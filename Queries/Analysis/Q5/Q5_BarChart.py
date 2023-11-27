import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

csvFile = pd.read_csv('./Query/Q5/Q5.csv', sep=',', names=['RepositoryName','NumOfCommits', 'CommitFrequency (days)'],
                      skiprows=1)

dfCsv = pd.DataFrame(csvFile)

modifiedFilesDf = dfCsv["CommitFrequency (days)"].dropna().to_numpy()

barChartData = pd.cut(modifiedFilesDf, [0,1,3,7,14,30,59,np.inf], labels=['[0,1]','[2,3]','[4,7]','[8,14]','[15,30]','[31,59]','[60+]'])
barChart = barChartData.value_counts().plot.bar(color="green")

plt.xticks(rotation=0)
plt.title("Average Frequency Of Commitments For Repository")
plt.xlabel('Commit Frequency (Days)') 
plt.ylabel('Number Of Repositories')

barChart.bar_label(barChart.containers[0])

plt.show()


