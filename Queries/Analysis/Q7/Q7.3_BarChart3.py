import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

dfRepositoryMetadata = pd.read_csv('./Queries/Query/Commits_metadata_p1.csv', sep=',',
                                   names=['RepositoryName', 'CommitName', 'Authors', 'CommitId', 'ModifiedFiles', 'DateTime'])

mergeKeywords = ["Merge", "merge", "merged", "Merged"]

# Creare un dizionario per contenere i conteggi delle occorrenze per ogni repository
mergeCounts = {'RepositoryName': []}
for keyword in mergeKeywords:
    mergeCounts[keyword] = []

# Ciclo sulle parole chiave di merge
for keyword in mergeKeywords:
    commitNames = dfRepositoryMetadata["CommitName"].fillna("").astype(str)
    modifiedFiles = dfRepositoryMetadata["ModifiedFiles"].fillna("").astype(str)

    occurrencesInNames = commitNames.apply(lambda x: keyword in x)
    occurrencesInFilePaths = modifiedFiles.apply(lambda x: keyword in x)

    # Calcola il totale delle occorrenze per ogni repository
    totalOccurrences = occurrencesInNames.groupby(dfRepositoryMetadata['RepositoryName']).sum()

    # Aggiungi i conteggi al dizionario
    mergeCounts[keyword] = totalOccurrences.tolist()

# Aggiungi i nomi dei repository al dizionario
mergeCounts['RepositoryName'] = dfRepositoryMetadata['RepositoryName'].unique().tolist()

# Creare un DataFrame dai conteggi
dfMergeOccurrences = pd.DataFrame(mergeCounts)

# Calcolare il numero totale di occorrenze per ogni repository
dfMergeOccurrences['TotalOccurrences'] = dfMergeOccurrences.iloc[:, 1:].sum(axis=1)

# Creare le categorie per il bar chart
# 0, 1-10
categories = pd.cut(dfMergeOccurrences['TotalOccurrences'], [0, 1, 10, 25, 50, 100, 200, 300, 400, 500, np.inf],
                    labels=['[0]','[1,10]', '[11,25]', '[26,50]', '[51,100]', '[101,200]', '[201,300]', '[301,400]', '[401,500]', '[500+]'])

# Plot del bar chart
barChart = categories.value_counts().sort_index().plot.bar(color="green", yticks=[100, 250, 500, 1000, 1500, 2000, 2500])

plt.xticks(rotation=45)
#plt.title("Distribution of occurrences of the word 'merge' for each repository")
plt.xlabel('Number of Merge')
plt.ylabel('Number of Repository')

# Aggiungi le etichette alle barre
barChart.bar_label(barChart.containers[0])

plt.show()
