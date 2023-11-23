# DevOpsMobileApllicationDevelopment

In the file RepositoryList + ManualAnalysis, all the data on the repositories are taken into account, including a sample of 2000 commits, and a commit hash spreadsheet.

In the folder First Analysis there are the following files:
* 01 - DataCleaning.

This script in Python performs a series of operations for cleaning and manipulating data in a CSV file. Initially, the necessary libraries such as Pandas, Path from pathlib, math, matplotlib (with alias plt), and NumPy are imported. Next, the code reads a CSV file called 'SampleCommits.csv' and creates a Dataframe called df. Once the Dataframe has been created, some columns that appear to be irrelevant for the subsequent analysis are removed.

The code then deals with the handling of missing values (NaN), checking for the presence of these values in the Dataframe, and deleting the rows that contain them. In addition, rows containing the string 'NO LABEL' in the 'CATEGORIES:' column are filtered out.

Finally, the cleaned DataFrame is saved in a new CSV file called 'DataframeCleaned.csv'. This script is designed to prepare and clean the data, making it more suitable for future analysis or processing.

* 02 - GeneralMachineLearningMethods+Accuracy

This Python script represents an approach to the automatic classification of software commit messages using machine learning techniques. The main goal is to develop and evaluate different classification models to automatically assign labels to commit categories. This task is crucial for better understanding the nature of changes within a software project and for facilitating the management and analysis of large volumes of commits.

The script begins with the import of necessary libraries, such as Pandas for data manipulation and scikit-learn for building and evaluating machine learning models. Subsequently, a CSV file containing previously cleaned commit data is read, including information about commit messages and their respective categories.

The data preparation process involves splitting messages and labels into training and test sets. This allows the evaluation of the model's performance on previously unseen data.

A crucial part of the script involves the definition and training of various classification models, each designed to learn distinct patterns from the training data. Included models are Support Vector Machine, Decision Tree, Naive Bayes, Stochastic Gradient Descent, Neural Network, and Random Forest.

The training process uses a pipeline that combines text vectorization (CountVectorizer), Tf-idf transformation (TfidfTransformer), and the classifier itself. This pipeline enables a smooth data flow through the different stages of the learning process.

To assess the effectiveness of each model, cross-validation is performed on test data, providing reliable estimates of the model's performance. Cross-validation results are printed, highlighting the mean accuracy and standard deviation for each model.

In conclusion, the script provides a comprehensive overview of the training and evaluation process for text classification models to automatically assign categories to software commit messages. This methodology is essential for improving efficiency and accuracy in managing changes within a software project.

* 03 - DataCleaning_DataSet1

This Python script revolves around the preparation and cleaning of commit metadata, extracted from a CSV file named 'Commits_metadata_p1.csv'. The primary objective is to streamline the dataset for subsequent analysis by removing unnecessary columns and handling missing values.

Following the import of essential libraries, such as Pandas for data manipulation and Path from pathlib for file path management, the script proceeds to read the commit metadata into a Pandas Dataframe named `dataframe`.

To refine the dataset, certain columns deemed extraneous for the analysis are dropped. These columns include 'RepositoryName', 'Authors', 'CommitId', 'ModifiedFiles', and 'DateTime'. The resulting Dataframe is then displayed to provide insight into the modifications.

Subsequently, the script checks for the presence of missing values (NaN) within the dataset. Specifically, it examines the 'CommitName' column for NaN values, ensuring a thorough assessment of potential data gaps.

To enhance data integrity, rows containing NaN values are removed from the DataFrame. This step is crucial for maintaining consistency and reliability in subsequent analyses.

A confirmation check is then conducted to verify the successful removal of NaN values from the dataset. This ensures that the cleaned Dataframe is ready for further exploration.

As part of the exploratory process, the script prints the data types of each column in the DataFrame. This provides valuable information about the nature of the features within the dataset.

Lastly, the cleaned DataFrame is saved to a new CSV file named 'Dataframe1_AllCommits'. This step facilitates ease of use for subsequent analyses, allowing researchers and data scientists to readily access and employ the refined commit metadata.

In summary, the script exemplifies a fundamental data-cleaning process, emphasizing the importance of preparing datasets to ensure accuracy and reliability in subsequent analytical endeavors.

* 04 - DataCleaning_DataSet2

This Python script is dedicated to refining and preparing to commit metadata sourced from the 'Commits_metadata_p2.csv' file. The process involves a series of steps aimed at optimizing the dataset for subsequent analyses.

To begin, the script imports essential libraries, such as Pandas for data manipulation and Path from pathlib for managing file paths. The commit metadata is then read into a Pandas DataFrame named `dataframe`. Notably, custom names are assigned to the columns: 'RepositoryName', 'CommitName', 'Authors', 'CommitId', 'ModifiedFiles', and 'DateTime'.

The original structure and content of the Dataframe are printed, providing an initial overview of the dataset. Subsequently, certain columns ('RepositoryName', 'Authors', 'CommitId', 'ModifiedFiles', 'DateTime') are deemed extraneous and are consequently dropped from the Dataframe.

The Dataframe is printed again post-column removal, offering insight into the refined structure. Following this, the script checks for the presence of NaN values within the dataset. Specifically, it examines the 'CommitName' column for any NaN entries.

To enhance data integrity, rows containing NaN values are removed from the Dataframe. A verification step is implemented to ensure the successful elimination of NaN values, paving the way for a cleaner and more reliable dataset.

The data types of each column within the Dataframe are then printed, offering valuable insights into the nature of the dataset's features.

Lastly, the cleaned Dataframe is saved to a new CSV file named 'Dataframe2_AllCommits'. This step is crucial for preserving the cleaned dataset and facilitating easy access for subsequent analyses.

In summary, this script underscores a meticulous data cleaning process, emphasizing the removal of unnecessary columns and the handling of missing values to ensure the dataset's readiness for comprehensive analysis.

* 05 - SDG_Dataframe1

This Python script focuses on implementing a machine-learning model for text classification within commit messages.

The code begins by importing key libraries such as NumPy, Pandas, and scikit-learn, essential for data manipulation and applying machine learning techniques. Next, commit metadata is loaded from a CSV file ('DataframeCleaned.csv') into a Pandas DataFrame named `df`, providing an initial glimpse into its content.

The dataset is then split into training and test sets using the `train_test_split` function from scikit-learn, a common practice in preparing data for machine learning model training. A text classification pipeline is constructed using scikit-learn's `Pipeline` class, involving text vectorization, TF-IDF transformation, and the use of a linear SGD classifier. The model is trained on the training data, and predictions are made on both the training and test data. Subsequently, the accuracies of the model on these sets are calculated and printed.

Finally, the trained model is applied to new data, specifically, commit names in another dataset ('Dataframe1_AllCommits.csv'). Predictions are then inserted into the corresponding DataFrame, and the modified result is saved in a new CSV file ('Dataframe1_AllCommits.csv').

Overall, the script demonstrates a comprehensive procedure from training a text classification machine learning model to applying it to new data.

* 06 - SDG_Dataframe2

In this Python script, the focus is on leveraging machine learning techniques for text classification, specifically applied to commit messages in a software development context. The script utilizes scikit-learn, a popular machine learning library, to implement a Support Vector Machine (SGDClassifier) for categorizing commit messages into predefined categories.

Firstly, the script reads cleaned commit data from a CSV file ('DataframeCleaned.csv') and displays the content. This dataset is then split into training and testing sets, a fundamental step in supervised machine learning.

Next, a text classification pipeline is established. This pipeline involves the conversion of text into numerical features using `CountVectorizer`, followed by the transformation of these features into a TF-IDF representation using `TfidfTransformer`. The actual classification is performed by an SGDClassifier.

The model is trained on the training set, and predictions are made on both the training and testing sets. Accuracy scores are calculated to evaluate the performance of the model on these sets.

Subsequently, the script introduces a new dataset from another CSV file ('Dataframe2_AllCommits.csv') and predicts categories for the commit names using the trained text classification model. These predictions are then inserted into the DataFrame, and the modified DataFrame is saved to a new CSV file ('Dataframe2_AllCommits.csv').

In essence, this script showcases the application of machine learning to automate the categorization of commit messages, enabling more efficient management and analysis of software development activities. The use of a pipeline and the SGDClassifier reflects a thoughtful approach to text classification, demonstrating the versatility and power of machine learning in the context of software engineering.

*
*
*
*
*
*
*
