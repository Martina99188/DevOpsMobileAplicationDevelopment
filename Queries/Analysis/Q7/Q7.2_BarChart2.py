import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

csvFile = pd.read_csv('./Query/Q7.2/Q7.2.csv', sep=',', names=['RepositoryName','NumOfPullRequests', 'NumOfForks'],
                      skiprows=1)

dfCsv = pd.DataFrame(csvFile)

pullRequestDf = dfCsv["NumOfForks"].to_numpy()

barChartData = pd.cut(pullRequestDf, [0,10,25,50,100,200,300,400,500,np.inf], labels=['[0,10]','[11,25]','[26,50]','[51,100]','[101,200]','[201,300]','[301,400]','[401,500]','[500+]'] )
barChart = barChartData.value_counts().plot.bar(color="green" ,yticks=[100,250,500,1000,1500,2000,2500])

plt.xticks(rotation=45)
plt.title("Distribution Of Forks For Repositories")
plt.xlabel('Number Of Forks') 
plt.ylabel('Number Of Repositories')

barChart.bar_label(barChart.containers[0])

plt.show()