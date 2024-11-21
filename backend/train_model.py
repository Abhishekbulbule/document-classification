
# 76 percent
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
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

# Create a pipeline with TfidfVectorizer and VotingClassifier
pipeline = make_pipeline(
    TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_df=0.75, min_df=2),
    VotingClassifier(
        estimators=[
            ('lr', LogisticRegression(max_iter=1000)),
            ('svc', SVC(probability=True, kernel='linear')),
            ('nb', MultinomialNB())
        ],
        voting='soft'
    )
)

# Train the model
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

print(f'Accuracy: {accuracy*100}')
print(f'Precision: {precision*100}')
print(f'Recall: {recall*100}')
print(f'F1-Score: {f1*100}')
print(f'Confusion Matrix:\n{conf_matrix}')


# 76% accuracy
# from sklearn.datasets import fetch_20newsgroups
# from sklearn.model_selection import train_test_split
# from sklearn.pipeline import make_pipeline
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.linear_model import LogisticRegression
# from sklearn.svm import SVC
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.ensemble import VotingClassifier
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# # Load the 20 Newsgroups dataset
# categories = ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware',
#               'comp.sys.mac.hardware', 'comp.windows.x', 'misc.forsale', 'rec.autos', 'rec.motorcycles',
#               'rec.sport.baseball', 'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space',
#               'soc.religion.christian', 'talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc',
#               'talk.religion.misc']

# # Load data with preprocessing
# newsgroups_train = fetch_20newsgroups(subset='train', categories=categories, 
#                                        remove=('headers', 'footers', 'quotes'))
# newsgroups_test = fetch_20newsgroups(subset='test', categories=categories, 
#                                       remove=('headers', 'footers', 'quotes'))

# # Split data
# X_train, X_test, y_train, y_test = train_test_split(
#     newsgroups_train.data, 
#     newsgroups_train.target, 
#     test_size=0.2, 
#     random_state=42
# )

# # Create pipeline with voting classifier using probability-based classifiers
# pipeline = make_pipeline(
#     TfidfVectorizer(
#         stop_words='english',
#         ngram_range=(1, 2),
#         max_df=0.7,
#         min_df=2
#     ),
#     VotingClassifier(
#         estimators=[
#             ('lr', LogisticRegression(max_iter=5000)),
#             ('svm', SVC(probability=True, kernel='linear')),
#             ('nb', MultinomialNB())
#         ],
#         voting='soft'
#     )
# )

# # Train the model
# pipeline.fit(X_train, y_train)

# # Predict and evaluate
# y_pred = pipeline.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)
# precision = precision_score(y_test, y_pred, average='weighted')
# recall = recall_score(y_test, y_pred, average='weighted')
# f1 = f1_score(y_test, y_pred, average='weighted')

# print(f'Accuracy: {accuracy*100:.2f}%')
# print(f'Precision: {precision*100:.2f}%')
# print(f'Recall: {recall*100:.2f}%')
# print(f'F1-Score: {f1*100:.2f}%')

# from sklearn.datasets import fetch_20newsgroups
# from sklearn.model_selection import train_test_split
# from sklearn.pipeline import make_pipeline
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
# import joblib

# # Load the 20 Newsgroups dataset
# categories = ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware',
#               'comp.sys.mac.hardware', 'comp.windows.x', 'misc.forsale', 'rec.autos', 'rec.motorcycles',
#               'rec.sport.baseball', 'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space',
#               'soc.religion.christian', 'talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc',
#               'talk.religion.misc']

# newsgroups_train = fetch_20newsgroups(subset='train', categories=categories, remove=('headers', 'footers', 'quotes'))
# newsgroups_test = fetch_20newsgroups(subset='test', categories=categories, remove=('headers', 'footers', 'quotes'))

# X_train, X_test, y_train, y_test = train_test_split(newsgroups_train.data, newsgroups_train.target, test_size=0.2, random_state=42)

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

# print(f'Accuracy: {accuracy*100}')
# print(f'Precision: {precision*100}')
# print(f'Recall: {recall*100}')
# print(f'F1-Score: {f1*100}')
# print(f'Confusion Matrix:\n{conf_matrix}')
