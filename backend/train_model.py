from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib

# Load the 20 Newsgroups dataset
categories = ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware',
              'comp.sys.mac.hardware', 'comp.windows.x', 'misc.forsale', 'rec.autos', 'rec.motorcycles',
              'rec.sport.baseball', 'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space',
              'soc.religion.christian', 'talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc',
              'talk.religion.misc']

newsgroups_train = fetch_20newsgroups(subset='train', categories=categories, remove=('headers', 'footers', 'quotes'))
newsgroups_test = fetch_20newsgroups(subset='test', categories=categories, remove=('headers', 'footers', 'quotes'))

X_train, X_test, y_train, y_test = train_test_split(newsgroups_train.data, newsgroups_train.target, test_size=0.2, random_state=42)

# Create and train the model
pipeline = make_pipeline(TfidfVectorizer(stop_words='english'), MultinomialNB())
pipeline.fit(X_train, y_train)

# Save the model
joblib.dump(pipeline, 'model.joblib')

# Evaluate the model
y_pred = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1-Score: {f1}')
print(f'Confusion Matrix:\n{conf_matrix}')


# import os
# from sklearn.model_selection import train_test_split
# from sklearn.pipeline import make_pipeline
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
# import joblib

# # Load dataset from local directory
# def load_files_from_directory(directory):
#     data = []
#     target = []
#     target_names = []
#     for category in os.listdir(directory):
#         category_path = os.path.join(directory, category)
#         if os.path.isdir(category_path):
#             file_path = os.path.join(category_path, category + ".txt")
#             if os.path.isfile(file_path):
#                 with open(file_path, 'r', encoding='latin1') as file:
#                     content = file.read()
#                     if len(content.strip()) > 0:  # Check if content is not empty
#                         data.append(content)
#                         target.append(category)  # Assign category as target label
#                         target_names.append(category)
#                         print(f"Loaded file: {file_path}, Content length: {len(content)}")  # Print content length
#             else:
#                 print(f"File not found: {file_path}")  # Print statement for missing files
#     return data, target, target_names

# data, target, target_names = load_files_from_directory('./archive')
# print(f"Total files loaded: {len(data)}")  # Print total number of files loaded

# # Check data distribution
# from collections import Counter
# print(f"Data distribution: {Counter(target)}")

# X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42)

# # Create and train the model
# pipeline = make_pipeline(TfidfVectorizer(stop_words='english'), MultinomialNB())
# pipeline.fit(X_train, y_train)

# # Save the model
# joblib.dump(pipeline, 'model.joblib')

# # Evaluate the model
# y_pred = pipeline.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)
# precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
# recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
# f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
# conf_matrix = confusion_matrix(y_test, y_pred)

# print(f'Accuracy: {accuracy}')
# print(f'Precision: {precision}')
# print(f'Recall: {recall}')
# print(f'F1-Score: {f1}')
# print(f'Confusion Matrix:\n{conf_matrix}')

# import os
# from sklearn.model_selection import train_test_split
# from sklearn.pipeline import make_pipeline
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.linear_model import LogisticRegression
# import joblib

# # Load dataset from local directory
# def load_files_from_directory(directory):
#     data = []
#     target = []
#     target_names = []
#     for category in os.listdir(directory):
#         category_path = os.path.join(directory, category)
#         if os.path.isdir(category_path):
#             for filename in os.listdir(category_path):
#                 if filename.endswith(".txt"):
#                     with open(os.path.join(category_path, filename), 'r', encoding='latin1') as file:
#                         data.append(file.read())
#                         target.append(category)  # Assign category as target label
#                         target_names.append(filename)
#     return data, target, target_names

# data, target, target_names = load_files_from_directory('./archive')
# X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42)

# # Create and train the model
# pipeline = make_pipeline(TfidfVectorizer(stop_words='english'), LogisticRegression())
# pipeline.fit(X_train, y_train)

# # Save the model
# joblib.dump(pipeline, 'model.joblib')
