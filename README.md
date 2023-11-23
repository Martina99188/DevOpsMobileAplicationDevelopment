# DevOpsMobileApllicationDevelopment

In the file RepositoryList + ManualAnalysis there are all the data on the repositories taken into account, a sample of 2000 commits and a commit hash spreedsheet.

In the folder First Analysis there are the following files:
* 01 - DataCleaning.

This script in Python performs a series of operations for cleaning and manipulating data in a CSV file. Initially, the necessary libraries such as Pandas, Path from pathlib, math, matplotlib (with alias plt) and NumPy are imported. Next, the code reads a CSV file called 'SampleCommits.csv' and creates a DataFrame called df. Once the DataFrame has been created, some columns that appear to be irrelevant for the subsequent analysis are removed.
The code then deals with the handling of missing values (NaN), checking for the presence of these values in the DataFrame and deleting the rows that contain them. In addition, rows containing the string 'NO LABEL' in the 'CATEGORIES:' column are filtered out.
Finally, the cleaned DataFrame is saved in a new CSV file called 'DataframeCleaned.csv'. In summary, this script is designed to prepare and clean the data, making it more suitable for future analysis or processing.
*
*
*
*
*
*
*
*
*
*
*
*
