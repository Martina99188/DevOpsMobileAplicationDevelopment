import pandas as pd
import csv
from csv import writer

csvMetadati = r'../Query/RQ1.2/RQ1.2.csv'
csvAnalisiFinale = r'../Analisi/RQ1.2/fogliDevOps/AnalisiFinale.csv'
csvHashCommit = r'../Analisi/RQ1.2/fogliDevOps/HashCommit.csv'

metadati = pd.read_csv(csvMetadati, sep=',', names=['RepositoryName','CommitName','Authors','CommitId','ModifiedFiles','DateTime'])
AnalisiFinale = pd.read_csv(csvAnalisiFinale, sep=',', names=['CommitName','CommitLabel'])
HashCommit = pd.read_csv(csvHashCommit, sep=',', names=['CommitName','CommitId'])

for i, commit in metadati.iterrows():

    currentCommitId = commit["CommitId"]

    commitInHashCommit = HashCommit.query("CommitId == @currentCommitId")

    if(len(commitInHashCommit) > 0 ):

        commitHashName = commitInHashCommit['CommitName'].values[0]

        commitInAnalisiFinale = AnalisiFinale.query('CommitName == @commitHashName')

        if(len(commitInAnalisiFinale) > 0 and commitInAnalisiFinale['CommitLabel'].values[0] == "RELEASE"):

            metadata = commit
            with open('CorrispondenzeDeploy.csv', 'a', encoding='UTF8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(metadata)