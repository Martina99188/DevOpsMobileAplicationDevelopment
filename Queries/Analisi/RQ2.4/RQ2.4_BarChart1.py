import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

csvFile = pd.read_csv('./Query/RQ2.4/RQ2.4.csv', sep=',', names=['RepositoryName','NumOfContributors', 'Contributor', '%OfContribution'],
                      skiprows=1)

dfCsv = pd.DataFrame(csvFile)

repositoryContributorsDf = dfCsv[["RepositoryName","NumOfContributors"]]

numOfContributors = repositoryContributorsDf.drop_duplicates(subset=['RepositoryName'], keep='first')

numOfContributors = numOfContributors['NumOfContributors'].to_numpy()

barChartData = pd.cut(numOfContributors, [0,3,5,10,15,30,50,100, np.inf], labels=['[0,3]','[4,5]','[6,10]','[11,15]','[16-30]','[31-50]','[51,100]','[100+]'], )
barChart = barChartData.value_counts().plot.bar(color="green")

plt.xticks(rotation=0)
plt.title("Distribution of the Number of Contributors Per Repository")
plt.xlabel('Number Of Contributors') 
plt.ylabel('Number Of Repositories')

barChart.bar_label(barChart.containers[0])

plt.show()


