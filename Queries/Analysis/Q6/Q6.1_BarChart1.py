import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

csvFile = pd.read_csv('./Queries/Query/Q6/Q6.csv', sep=',', names=['RepositoryName','NumOfContributors', 'Contributor', '%OfContribution'],
                      skiprows=1)

dfCsv = pd.DataFrame(csvFile)

repositoryContributorsDf = dfCsv[["RepositoryName","NumOfContributors"]]

numOfContributors = repositoryContributorsDf.drop_duplicates(subset=['RepositoryName'], keep='first')

numOfContributors = numOfContributors['NumOfContributors'].to_numpy()

barChartData = pd.cut(numOfContributors, [0,2,5,9,10,np.inf], labels=['[0]','[1,2]','[3,5]','[6,9]','[10+]'])
barChart = barChartData.value_counts().plot.bar(color="green")

plt.xticks(rotation=0)
plt.title("Distribution of the Number of Contributors For Repository")
plt.xlabel('Number Of Contributors') 
plt.ylabel('Number Of Repositories')

barChart.bar_label(barChart.containers[0])

plt.show()


