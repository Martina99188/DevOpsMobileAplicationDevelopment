# DevOpsMobileAplicationDevelopment

In the folder **ManualAnalysis/** there is the following file:
* **RepositoryList + ManualAnalysis.csv**.
There is all the data on the extracted repositories, a sample of 2000 commits, and a spreadsheet of commit hashes.

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

In the folder **Queries** there are 2 folders. The folder called **Metadata** has the following files:

* **Releases_metadata.csv**.

This file contains the name of the repository, the release, the release date, the author, and the last commit ID. 

* **Repository_metadata.csv**.

This file contains the name of the repository, the number of stars, the number of forks, the number of commits, the number of releases, the number of contributors, the number of issues, and the number of pull requests. All this data will allow graphs to be created as required.


The second folder is called **Analysis** and it has the following folders:

* **Q1** that contains:
  * **sheetsDevOps** folder:
 
    * **FinalAnalysis.csv**. The file contains the commit and the categories associated with each commit.
    * **HashCommit.csv**. The file contains the commit and ID associated with each commit.

  * **compareMetadata**

This Python code utilizes the `pandas` and `csv` libraries to process and filter data from multiple CSV files. 

The code reads three CSV files containing metadata, final analysis results, and commit hashes. It creates dataframes using the `pandas` library.

It then iterates through the rows of the 'metadata' dataframe, checking if the current commit ID is present in both the 'HashCommit' and 'FinalAnalysis' dataframes.

If the commit is found in both dataframes and is labeled as a "RELEASE" in the 'FinalAnalysis' dataframe, the relevant metadata is selected and written to a new CSV file named 'CorrespondencesDeploy.csv'.

The code is essentially filtering and extracting specific information based on conditions related to commit IDs and labels, and the selected data is saved in a separate CSV file for further analysis or reference.

  * **CorrespondencesDeploy**

The file contains all information about the repository and the commits considered for the Deploy phase.

  * **CorrespondencesTest**

The file contains all information about the repository and the commits considered for the Test phase.

* **Q2**
  * **Q2_BoxPlot.py**

This Python code utilizes the `matplotlib`, `numpy`, `pandas`, and `statistics` libraries to analyze and visualize the distribution of time distances from a previous commit across a set of software repositories.

The process begins with reading a CSV file containing data about repository commits. The data is structured into a dataframe using the `pandas` library.

Subsequently, time distances from previous commits are processed, disregarding zero values, and the results are stored in a NumPy array.

The code iterates through unique repositories, calculating the median time distances from previous commits for each repository and saving the results in a list.

Using the `matplotlib` library, a boxplot is created to represent the distribution of time distances, including the median for each repository. Customizations are made to the plot, such as removing outliers and adding labels.

Finally, the boxplot is displayed, clearly illustrating the distribution of time distances from a previous commit for each repository, with a focus on the median and other relevant details.
  
  * **Q2.1.png**

Commits’ time distance from the previous release.

  * **Q2.2.png**

Commits’ time distance from the next release.

* **Q3** that contains:

  * **Q3_BoxPlot.py**

The code reads a CSV file named 'RQ2.1.csv' containing data on repository names, the number of releases, and release frequencies in days. The data is organized into a dataframe using the pandas library.

The release frequencies are extracted from the dataframe, and zero values are replaced with NaN (Not a Number) to avoid distortion in the boxplot. The cleaned data is then converted to a numpy array.

A boxplot is created using matplotlib to display the distribution of release frequencies. The boxplot does not include outliers for a clearer representation.

The plot is customized with a title, y-axis label, and x-axis tick labels. The median, whiskers, and caps of the boxplot are annotated with their respective values for additional information.

Finally, the boxplot is displayed using plt.show(). The visualization provides an overview of the release frequency distribution for the analyzed repositories.

  * **Q3.png**

New release frequency grouped by the repository.

* **Q4** that contains:

  * **Q4_BarChart.py**

The code reads a CSV file named 'RQ2.2.csv' containing data on repository names, commit IDs, and the number of modified files per commit. The data is organized into a dataframe using the pandas library.

The number of modified files is extracted from the dataframe, and NaN (Not a Number) values are dropped. The cleaned data is then converted to a numpy array.

A bar chart is created using matplotlib and the pd.cut function to categorize the number of modified files into bins. The chart displays the count of commits in each bin.

The x-axis tick labels, title, and axis labels are customized for clarity. The bar chart labels each bar with the respective count of commits.

Finally, the bar chart is displayed using plt.show(). The visualization provides insight into the distribution of the number of modified files per commit across the analyzed repositories.

  * **Q4.png**

Number of modified files per commit.

* **Q5** that contains:

  * **Q5_BarChart.py**

The code reads a CSV file named 'RQ2.3.csv' containing data on repository names, the number of commits, and commit frequencies in days. The data is organized into a dataframe using the pandas library.

The commit frequencies are extracted from the dataframe, and NaN (Not a Number) values are dropped. The cleaned data is then converted to a numpy array.

A bar chart is created using matplotlib and the pd.cut function to categorize the commit frequencies into bins. The chart displays the count of repositories in each bin.

The x-axis tick labels, title, and axis labels are customized for clarity. Each bar in the chart is labeled with the count of repositories it represents.

Finally, the bar chart is displayed using plt.show(). The visualization provides an overview of the distribution of average commit frequencies across the analyzed repositories.

  * **Q5.png**

Commits’ average frequency grouped by repository.

* **Q6** that contains:

  * **Q6_BarChart1.py**

The code reads a CSV file named 'RQ2.4.csv' containing data on repository names, the number of contributors, individual contributor names, and the percentage of contribution. The data is organized into a dataframe using the pandas library.

The dataframe is then modified to focus on the number of contributors per repository, removing duplicate entries for each repository.

A bar chart is created using matplotlib and the pd.cut function to categorize the number of contributors into bins. The chart displays the count of repositories in each bin.

The x-axis tick labels, title, and axis labels are customized for clarity. Each bar in the chart is labeled with the count of repositories it represents.

Finally, the bar chart is displayed using plt.show(). The visualization provides insight into the distribution of the number of contributors per repository across the analyzed dataset.

  * **Q6.1.png**

The number of contributors grouped by the repository.

  * **Q6_BarChart2.py***

The code reads a CSV file named 'RQ2.4.csv' containing data on repository names, the number of contributors, individual contributor names, and the percentage of contribution. The data is organized into a dataframe using the pandas library.

The contribution percentages are extracted from the dataframe and converted to a numpy array.

A bar chart is created using matplotlib and the pd.cut function to categorize the contribution percentages into bins. The chart displays the count of contributors in each bin.

The x-axis tick labels, title, and axis labels are customized for clarity. Each bar in the chart is labeled with the count of contributors it represents.

Finally, the bar chart is displayed using plt.show(). The visualization provides an overview of the distribution of contribution percentages across the analyzed contributors.

  * **Q6.2.png**

Percentage of contribution.

* **Q7** that contains:

  * **Q7_BarChart1.py**

The code reads a CSV file named 'RQ2.5.csv' containing data on repository names, the number of pull requests, and the number of forks. The data is organized into a dataframe using the pandas library.

The number of pull requests is extracted from the dataframe and converted to a numpy array.

A bar chart is created using matplotlib and the pd.cut function to categorize the number of pull requests into bins. The chart displays the count of repositories in each bin.

The x-axis tick labels, title, and axis labels are customized for clarity. The y-axis ticks are manually set to specific values to better represent the data.

Finally, the bar chart is displayed using plt.show(). The visualization provides insight into the distribution of the number of pull requests across the analyzed repositories.

  * **Q7.1.png**

Number of pull requests per repository.

  * **Q7_BarChart2.py**

The code reads a CSV file named 'RQ2.5.csv' containing data on repository names, the number of pull requests, and the number of forks. The data is organized into a dataframe using the pandas library.

The number of forks is extracted from the dataframe and converted to a numpy array.

A bar chart is created using matplotlib and the pd.cut function to categorize the number of forks into bins. The chart displays the count of repositories in each bin.

The x-axis tick labels, title, and axis labels are customized for clarity. The y-axis ticks are manually set to specific values to better represent the data.

Finally, the bar chart is displayed using plt.show(). The visualization provides insight into the distribution of the number of forks across the analyzed repositories.
 
  * **Q7.2.png**

Number of forks per repository.

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

In the folder **AutomaticAnalysis** there are the following files:

* **01 - DataCleaning.py**.

This Python script revolves around the reading and cleaning of data from a CSV file named 'NewSample.csv'. The process involves importing essential libraries such as Pandas for data manipulation and utilizing the Path module from pathlib for handling file paths.

The script proceeds to read the data from the CSV file into a Pandas dataframe named `df`, specifying a semicolon (';') as the separator.

Following data import, the script checks for the presence of NaN (Not a Number) values in the dataframe to ensure data integrity.

Finally, the cleaned dataframe is saved to a new CSV file named 'NewDF_cleaned_AllCategories.csv'. The saved file is organized within the 'sample_data' directory.

In essence, this script streamlines the dataset by handling missing values and provides a cleaned version for potential future analysis.

* **02 - GeneralMachineLearningMethods + Accuracy.py**.

This Python script is an engineering endeavor dedicated to automating the categorization of commit messages. By integrating essential libraries and constructing a text classification pipeline, the primary goal is to explore and evaluate the effectiveness of various machine learning models. 

From Support Vector Machines to Neural Networks, the focus is on gaining an in-depth understanding of the performance of each model in the intricate task of automatically assigning categories to commit messages. In this context, the script stands as a comprehensive exploration of text classification techniques applied to the specific realm of version control.

* **03 - GeneralMachineLearningMethods + Accuracy, Precision, Recall, F1-Score.py**.

This Python script is dedicated to the automated classification of commit messages using a variety of machine-learning models. Through the importation of essential libraries and the creation of a text classification pipeline, the code explores models such as Support Vector Machine, Decision Tree, Naive Bayes, Stochastic Gradient Descent, Neural Network, and Random Forest. 

The data is split into training and testing sets, and for each model, precision, recall, F1-score, and accuracy metrics are evaluated. This implementation provides a comprehensive overview of the performance of various machine learning models in the context of automatically categorizing commit messages.

* **04 - SVC & SDG + Accuracy, Precision, Recall, F1-Score, ROC, PR, K-Fold Cross-Validation, t-SNE.py**.

This Python script is designed to implement and evaluate text classification models, primarily focusing on multi-class classification tasks. Its functionalities include importing essential libraries, reading cleaned commit message data from a CSV file, separating messages and categories, splitting the data into training and testing sets, selecting and evaluating models such as Linear Support Vector Classifier (`LinearSVC`) and Stochastic Gradient Descent (`SGDClassifier`) using cross-validation. 

The code also implements the Binary Relevance approach for multi-class classification, computes Receiver Operating Characteristic (ROC) curves and Precision-Recall (PR) curves for each model, and displays these curves along with relevant metrics. 

Finally, it presents a t-SNE visualization for dimensionality reduction, representing data points on a 2D plot with distinct colors for each class and a customized legend.

* **05 - DataCleaning_Dataset1.py**.

This Python script focuses on cleaning and preparing data extracted from a CSV file containing information about software repository commits. It utilizes the Pandas library for data manipulation and pathlib for managing file paths. 

The key steps include removing unnecessary columns, handling missing values, specifically addressing NaN values in the 'CommitName' column, and saving the cleaned dataframe to a new CSV file ('dataframe1.csv'). 

The script ensures data consistency and usability for subsequent analysis by eliminating irrelevant information and managing missing values appropriately.

* **06 - DataCleaning_Dataset2.py**.

The script revolves around the processing and refinement of data related to software commit stored in a CSV file named 'Commits_metadata_p2.csv'. The primary actions include reading the CSV file using the Pandas library, removing unnecessary columns such as 'RepositoryName', 'Authors', 'CommitId', 'ModifiedFiles', and 'DateTime', and checking for the presence of missing values. 

Specifically focusing on the 'CommitName' column, the script identifies and handles any NaN (Not a Number) values, ensuring data integrity. After successfully eliminating missing values, the script saves the cleaned dataframe to a new CSV file named 'dataframe2.csv'. 

This systematic data-cleaning process aims to enhance the quality and usability of the commit data for subsequent analysis.

* **07 - SDG_Dataset1.py**.

The script is focused on implementing a text classification model using machine learning techniques. It begins by importing necessary libraries, such as NumPy, Pandas, and scikit-learn modules. The core dataset, presumably representing cleaned commit data, is read from a CSV file named 'NewDF_cleaned_AllCategories.csv'. The data is then split into training and testing sets using the train_test_split function.

The script employs a machine learning pipeline consisting of CountVectorizer, TfidfTransformer, and the SGDClassifier (Stochastic Gradient Descent) to vectorize the commit messages and train a classification model. The accuracy of the model is evaluated on both the training and testing sets.

Following the training phase, the model is applied to a different dataset ('dataframe1.csv') to predict categories for commit names. The predictions are inserted into the dataframe, and the modified dataframe is saved as 'dataframe1.csv'. Overall, the script demonstrates a basic text classification workflow using scikit-learn for commit message categorization.

* **08 - SDG_Dataset2.py**.

This script focuses on implementing a text classification model using scikit-learn, particularly the SGDClassifier (Stochastic Gradient Descent). The dataset, presumably representing cleaned commit data, is read from a CSV file named 'NewDF_cleaned_AllCategories.csv'. The data is then split into training and testing sets using the train_test_split function.

The core of the script involves the creation of a text classification pipeline, including processes such as Count Vectorization and TF-IDF transformation. The SGDClassifier is used as the classification model. The accuracy of the model is evaluated on both the training and testing sets.

After the training phase, the model is applied to a different dataset ('dataframe2.csv') to predict categories for commit names. The predictions are inserted into the dataframe, and the modified dataframe is saved as 'dataframe2.csv'. In summary, the script showcases a typical text classification workflow using scikit-learn for commit message categorization.

* **Commits_50x7.csv**.

In this file, 50 commits per category were extracted from the total two million commits. Subsequently, the veracity of the commits was checked manually, and finally, a table with the percentages of correct commits was defined.

* **NewDF_cleaned_AllCategories.csv**.

Once the Data Cleaning step has been performed, the new dataframe is saved in this file.

* **NewSample.csv**.

This file contains the new, refined sample, with the addition of new commits for the PLAN phase and the NO LABEL phase.
