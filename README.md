# DevOpsMobileAplicationDevelopment

In the folder **ManualAnalysis/** there is the following file:
* **RepositoryList + ManualAnalysis**.
There is all the data on the extracted repositories, a sample of 2000 commits, and a spreadsheet of commit hashes.

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

In the folder **Queries** there are 2 folders. The folder called Metadata has the following files:

* **Releases_metadata**.

This file contains the name of the repository, the release, the release date, the author, and the last commit ID. 

* **Repository_metadata**.

This file contains the name of the repository, the number of stars, the number of forks, the number of commits, the number of releases, the number of contributors, the number of issues and the number of pull requests. All this data will allow graphs to be created as required.


The second folder is called Analysis and it has the following files:

* ****.

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

In the folder **AutomaticAnalysis** there are the following files:

* **01 - DataCleaning**.

This Python script revolves around the reading and cleaning of data from a CSV file named 'NewSample.csv'. The process involves importing essential libraries such as Pandas for data manipulation and utilizing the Path module from pathlib for handling file paths.

The script proceeds to read the data from the CSV file into a Pandas dataframe named `df`, specifying a semicolon (';') as the separator.

Following data import, the script checks for the presence of NaN (Not a Number) values in the dataframe to ensure data integrity.

Finally, the cleaned dataframe is saved to a new CSV file named 'NewDF_cleaned_AllCategories.csv'. The saved file is organized within the 'sample_data' directory.

In essence, this script streamlines the dataset by handling missing values and provides a cleaned version for potential future analysis.

* **02 - GeneralMachineLearningMethods + Accuracy**.

This Python script is an engineering endeavor dedicated to automating the categorization of commit messages. By integrating essential libraries and constructing a text classification pipeline, the primary goal is to explore and evaluate the effectiveness of various machine learning models. 

From Support Vector Machines to Neural Networks, the focus is on gaining an in-depth understanding of the performance of each model in the intricate task of automatically assigning categories to commit messages. In this context, the script stands as a comprehensive exploration of text classification techniques applied to the specific realm of version control.

* **03 - GeneralMachineLearningMethods + Accuracy, Precision, Recall, F1-Score**.

This Python script is dedicated to the automated classification of commit messages using a variety of machine-learning models. Through the importation of essential libraries and the creation of a text classification pipeline, the code explores models such as Support Vector Machine, Decision Tree, Naive Bayes, Stochastic Gradient Descent, Neural Network, and Random Forest. 

The data is split into training and testing sets, and for each model, precision, recall, F1-score, and accuracy metrics are evaluated. This implementation provides a comprehensive overview of the performance of various machine learning models in the context of automatically categorizing commit messages.

* **04 - SVC & SDG + Accuracy, Precision, Recall, F1-Score, ROC, PR, K-Fold Cross-Validation, t-SNE**.

This Python script is designed to implement and evaluate text classification models, primarily focusing on multi-class classification tasks. Its functionalities include importing essential libraries, reading cleaned commit message data from a CSV file, separating messages and categories, splitting the data into training and testing sets, selecting and evaluating models such as Linear Support Vector Classifier (`LinearSVC`) and Stochastic Gradient Descent (`SGDClassifier`) using cross-validation. 

The code also implements the Binary Relevance approach for multi-class classification, computes Receiver Operating Characteristic (ROC) curves and Precision-Recall (PR) curves for each model, and displays these curves along with relevant metrics. 

Finally, it presents a t-SNE visualization for dimensionality reduction, representing data points on a 2D plot with distinct colors for each class and a customized legend.

* **05 - DataCleaning_Dataset1**.

This Python script focuses on cleaning and preparing data extracted from a CSV file containing information about software repository commits. It utilizes the Pandas library for data manipulation and pathlib for managing file paths. 

The key steps include removing unnecessary columns, handling missing values, specifically addressing NaN values in the 'CommitName' column, and saving the cleaned dataframe to a new CSV file ('dataframe1.csv'). 

The script ensures data consistency and usability for subsequent analysis by eliminating irrelevant information and managing missing values appropriately.

* **06 - DataCleaning_Dataset2**.

The script revolves around the processing and refinement of data related to software commit stored in a CSV file named 'Commits_metadata_p2.csv'. The primary actions include reading the CSV file using the Pandas library, removing unnecessary columns such as 'RepositoryName', 'Authors', 'CommitId', 'ModifiedFiles', and 'DateTime', and checking for the presence of missing values. 

Specifically focusing on the 'CommitName' column, the script identifies and handles any NaN (Not a Number) values, ensuring data integrity. After successfully eliminating missing values, the script saves the cleaned dataframe to a new CSV file named 'dataframe2.csv'. 

This systematic data-cleaning process aims to enhance the quality and usability of the commit data for subsequent analysis.

* **07 - SDG_Dataset1**.

The script is focused on implementing a text classification model using machine learning techniques. It begins by importing necessary libraries, such as NumPy, Pandas, and scikit-learn modules. The core dataset, presumably representing cleaned commit data, is read from a CSV file named 'NewDF_cleaned_AllCategories.csv'. The data is then split into training and testing sets using the train_test_split function.

The script employs a machine learning pipeline consisting of CountVectorizer, TfidfTransformer, and the SGDClassifier (Stochastic Gradient Descent) to vectorize the commit messages and train a classification model. The accuracy of the model is evaluated on both the training and testing sets.

Following the training phase, the model is applied to a different dataset ('dataframe1.csv') to predict categories for commit names. The predictions are inserted into the dataframe, and the modified dataframe is saved as 'dataframe1.csv'. Overall, the script demonstrates a basic text classification workflow using scikit-learn for commit message categorization.

* **08 - SDG_Dataset2**.

This script focuses on implementing a text classification model using scikit-learn, particularly the SGDClassifier (Stochastic Gradient Descent). The dataset, presumably representing cleaned commit data, is read from a CSV file named 'NewDF_cleaned_AllCategories.csv'. The data is then split into training and testing sets using the train_test_split function.

The core of the script involves the creation of a text classification pipeline, including processes such as Count Vectorization and TF-IDF transformation. The SGDClassifier is used as the classification model. The accuracy of the model is evaluated on both the training and testing sets.

After the training phase, the model is applied to a different dataset ('dataframe2.csv') to predict categories for commit names. The predictions are inserted into the dataframe, and the modified dataframe is saved as 'dataframe2.csv'. In summary, the script showcases a typical text classification workflow using scikit-learn for commit message categorization.
