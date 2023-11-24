import pandas as pd
import csv
from csv import writer

csvMetadata = r'../Query/RQ1.2/RQ1.2.csv'
csvFinalAnalysis = r'../Analysis/RQ1.2/fogliDevOps/FinalAnalysis.csv'
csvHashCommit = r'../Analysis/RQ1.2/fogliDevOps/HashCommit.csv'

metadata = pd.read_csv(csvMetadata, sep=',', names=['RepositoryName','CommitName','Authors','CommitId','ModifiedFiles','DateTime'])
FinalAnalysis = pd.read_csv(csvFinalAnalysis, sep=',', names=['CommitName','CommitLabel'])
HashCommit = pd.read_csv(csvHashCommit, sep=',', names=['CommitName','CommitId'])

for i, commit in metadata.iterrows():

    currentCommitId = commit["CommitId"]

    commitInHashCommit = HashCommit.query("CommitId == @currentCommitId")

    if(len(commitInHashCommit) > 0 ):

        commitHashName = commitInHashCommit['CommitName'].values[0]

        commitInFinalAnalysis = FinalAnalysis.query('CommitName == @commitHashName')

        if(len(commitInFinalAnalysis) > 0 and commitInFinalAnalysis['CommitLabel'].values[0] == "RELEASE"):

            metadata = commit
            with open('CorrespondencesDeploy.csv', 'a', encoding='UTF8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(metadata)