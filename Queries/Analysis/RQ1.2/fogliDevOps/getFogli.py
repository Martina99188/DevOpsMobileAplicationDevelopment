import pandas as pd
import csv
from csv import writer

Lists_path1 = r'../AnalisiFinale.csv' 
Lists_path2 = r'../CommitHash.csv' 

commits = pd.read_csv(Lists_path1, usecols=[0,1], skiprows=1)
dfCommits = pd.DataFrame(commits)

commitsList = dfCommits.values.tolist()

for commit in commitsList:
    urlIndex = commit[0].find('https://github.com/')
    commitName = commit[0][:urlIndex-3].replace(' ','').replace('\n','')
    commitLabel = commit[1]

    info = [commitName, commitLabel]
    with open('AnalisiFinale.csv', 'a', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(info)  
   
    
commits = pd.read_csv(Lists_path2, usecols=[0,2], skiprows=1)
dfCommits = pd.DataFrame(commits)

commitsList = dfCommits.values.tolist()

for commit in commitsList:
    commitName = str(commit[0]).replace(' ','').replace('\n','')
    commitId = commit[1]

    info = [commitName, commitId]
    with open('HashCommit.csv', 'a', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(info)  

