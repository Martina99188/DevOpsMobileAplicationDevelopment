# DevOpsMobileApllicationDevelopment

In the file **RepositoryList + ManualAnalysis**, all the data on the repositories are taken into account, including a sample of 2000 commits, and a commit hash spreadsheet.


In the folder **First Analysis** there are the following files:
* **01 - DataCleaning**.

This script in Python performs a series of operations for cleaning and manipulating data in a CSV file. Initially, the necessary libraries such as Pandas, Path from pathlib, math, matplotlib (with alias plt), and NumPy are imported. Next, the code reads a CSV file called 'SampleCommits.csv' and creates a dataframe called df. Once the dataframe has been created, some columns that appear to be irrelevant for the subsequent analysis are removed.

The code then deals with the handling of missing values (NaN), checking for the presence of these values in the dataframe, and deleting the rows that contain them. In addition, rows containing the string 'NO LABEL' in the 'CATEGORIES:' column are filtered out.

Finally, the cleaned dataframe is saved in a new CSV file called 'dataframeCleaned.csv'. This script is designed to prepare and clean the data, making it more suitable for future analysis or processing.

* **02 - GeneralMachineLearningMethods+Accuracy**.

This Python script represents an approach to the automatic classification of software commit messages using machine learning techniques. The main goal is to develop and evaluate different classification models to automatically assign labels to commit categories. This task is crucial for better understanding the nature of changes within a software project and for facilitating the management and analysis of large volumes of commits.

The script begins with the import of necessary libraries, such as Pandas for data manipulation and scikit-learn for building and evaluating machine learning models. Subsequently, a CSV file containing previously cleaned commit data is read, including information about commit messages and their respective categories.

The data preparation process involves splitting messages and labels into training and test sets. This allows the evaluation of the model's performance on previously unseen data.

A crucial part of the script involves the definition and training of various classification models, each designed to learn distinct patterns from the training data. Included models are Support Vector Machine, Decision Tree, Naive Bayes, Stochastic Gradient Descent, Neural Network, and Random Forest.

The training process uses a pipeline that combines text vectorization (CountVectorizer), Tf-idf transformation (TfidfTransformer), and the classifier itself. This pipeline enables a smooth data flow through the different stages of the learning process.

To assess the effectiveness of each model, cross-validation is performed on test data, providing reliable estimates of the model's performance. Cross-validation results are printed, highlighting the mean accuracy and standard deviation for each model.

In conclusion, the script provides a comprehensive overview of the training and evaluation process for text classification models to automatically assign categories to software commit messages. This methodology is essential for improving efficiency and accuracy in managing changes within a software project.

* **03 - DataCleaning_DataSet1**.

This Python script revolves around the preparation and cleaning of commit metadata, extracted from a CSV file named 'Commits_metadata_p1.csv'. The primary objective is to streamline the dataset for subsequent analysis by removing unnecessary columns and handling missing values.

Following the import of essential libraries, such as Pandas for data manipulation and Path from pathlib for file path management, the script proceeds to read the commit metadata into a Pandas dataframe named `dataframe`.

To refine the dataset, certain columns deemed extraneous for the analysis are dropped. These columns include 'RepositoryName', 'Authors', 'CommitId', 'ModifiedFiles', and 'DateTime'. The resulting dataframe is then displayed to provide insight into the modifications.

Subsequently, the script checks for the presence of missing values (NaN) within the dataset. Specifically, it examines the 'CommitName' column for NaN values, ensuring a thorough assessment of potential data gaps.

To enhance data integrity, rows containing NaN values are removed from the dataframe. This step is crucial for maintaining consistency and reliability in subsequent analyses.

A confirmation check is then conducted to verify the successful removal of NaN values from the dataset. This ensures that the cleaned dataframe is ready for further exploration.

As part of the exploratory process, the script prints the data types of each column in the dataframe. This provides valuable information about the nature of the features within the dataset.

Lastly, the cleaned dataframe is saved to a new CSV file named 'dataframe1_AllCommits'. This step facilitates ease of use for subsequent analyses, allowing researchers and data scientists to readily access and employ the refined commit metadata.

In summary, the script exemplifies a fundamental data-cleaning process, emphasizing the importance of preparing datasets to ensure accuracy and reliability in subsequent analytical endeavors.

* **04 - DataCleaning_DataSet2**.

This Python script is dedicated to refining and preparing to commit metadata sourced from the 'Commits_metadata_p2.csv' file. The process involves a series of steps aimed at optimizing the dataset for subsequent analyses.

To begin, the script imports essential libraries, such as Pandas for data manipulation and Path from pathlib for managing file paths. The commit metadata is then read into a Pandas dataframe named `dataframe`. Notably, custom names are assigned to the columns: 'RepositoryName', 'CommitName', 'Authors', 'CommitId', 'ModifiedFiles', and 'DateTime'.

The original structure and content of the dataframe are printed, providing an initial overview of the dataset. Subsequently, certain columns ('RepositoryName', 'Authors', 'CommitId', 'ModifiedFiles', 'DateTime') are deemed extraneous and are consequently dropped from the dataframe.

The dataframe is printed again post-column removal, offering insight into the refined structure. Following this, the script checks for the presence of NaN values within the dataset. Specifically, it examines the 'CommitName' column for any NaN entries.

To enhance data integrity, rows containing NaN values are removed from the dataframe. A verification step is implemented to ensure the successful elimination of NaN values, paving the way for a cleaner and more reliable dataset.

The data types of each column within the dataframe are then printed, offering valuable insights into the nature of the dataset's features.

Lastly, the cleaned dataframe is saved to a new CSV file named 'dataframe2_AllCommits'. This step is crucial for preserving the cleaned dataset and facilitating easy access for subsequent analyses.

In summary, this script underscores a meticulous data cleaning process, emphasizing the removal of unnecessary columns and the handling of missing values to ensure the dataset's readiness for comprehensive analysis.

* **05 - SDG_dataframe1**.

This Python script focuses on implementing a machine-learning model for text classification within commit messages.

The code begins by importing key libraries such as NumPy, Pandas, and scikit-learn, essential for data manipulation and applying machine learning techniques. Next, commit metadata is loaded from a CSV file ('dataframeCleaned.csv') into a Pandas dataframe named `df`, providing an initial glimpse into its content.

The dataset is then split into training and test sets using the `train_test_split` function from scikit-learn, a common practice in preparing data for machine learning model training. A text classification pipeline is constructed using scikit-learn's `Pipeline` class, involving text vectorization, TF-IDF transformation, and the use of a linear SGD classifier. The model is trained on the training data, and predictions are made on both the training and test data. Subsequently, the accuracies of the model on these sets are calculated and printed.

Finally, the trained model is applied to new data, specifically, commit names in another dataset ('dataframe1_AllCommits.csv'). Predictions are then inserted into the corresponding dataframe, and the modified result is saved in a new CSV file ('dataframe1_AllCommits.csv').

Overall, the script demonstrates a comprehensive procedure from training a text classification machine learning model to applying it to new data.

* **06 - SDG_dataframe2**.

In this Python script, the focus is on leveraging machine learning techniques for text classification, specifically applied to commit messages in a software development context. The script utilizes scikit-learn, a popular machine learning library, to implement a Support Vector Machine (SGDClassifier) for categorizing commit messages into predefined categories.

Firstly, the script reads cleaned commit data from a CSV file ('dataframeCleaned.csv') and displays the content. This dataset is then split into training and testing sets, a fundamental step in supervised machine learning.

Next, a text classification pipeline is established. This pipeline involves the conversion of text into numerical features using `CountVectorizer`, followed by the transformation of these features into a TF-IDF representation using `TfidfTransformer`. The actual classification is performed by an SGDClassifier.

The model is trained on the training set, and predictions are made on both the training and testing sets. Accuracy scores are calculated to evaluate the performance of the model on these sets.

Subsequently, the script introduces a new dataset from another CSV file ('dataframe2_AllCommits.csv') and predicts categories for the commit names using the trained text classification model. These predictions are then inserted into the dataframe, and the modified dataframe is saved to a new CSV file ('dataframe2_AllCommits.csv').

In essence, this script showcases the application of machine learning to automate the categorization of commit messages, enabling more efficient management and analysis of software development activities. The use of a pipeline and the SGDClassifier reflects a thoughtful approach to text classification, demonstrating the versatility and power of machine learning in the context of software engineering.


In the folder **Final Analysis** there are the following files:

* **01 - DataCleaning_NewSample**.

This Python script revolves around the reading and cleaning of data from a CSV file named 'NewSample.csv'. The process involves importing essential libraries such as Pandas for data manipulation and utilizing the Path module from pathlib for handling file paths.

The script proceeds to read the data from the CSV file into a Pandas dataframe named `df`, specifying a semicolon (';') as the separator.

Following data import, the script checks for the presence of NaN (Not a Number) values in the dataframe to ensure data integrity.

Finally, the cleaned dataframe is saved to a new CSV file named 'NewDF_cleaned_AllCategories.csv'. The saved file is organized within the 'sample_data' directory.

In essence, this script streamlines the dataset by handling missing values and provides a cleaned version for potential future analysis.

* **02 - GeneralMachineLearningMethods + Accuracy_NewSample**.

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

