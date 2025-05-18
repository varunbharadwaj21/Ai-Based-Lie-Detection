# -*- coding: utf-8 -*-
"""
Created on Sun Apr 13 16:41:39 2025

@author: VARUN
"""

#Use of TF-IDF
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Loading the dataset
file_path = "path.csv"
df = pd.read_csv(file_path)
df.head(11)

# Extracting the features (text) and labels (truth/lie)
X = df["statement"]  # The textual data
y = df["veracity"]   # The labels (0 = False, 1 = True)

#Split into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

# Converting text into numerical features using TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=5000)  # Using top 5000 words
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

print(f"TF-IDF matrix shape (train): {X_train_tfidf.shape}, (test): {X_test_tfidf.shape}")

#Support Vector Machine(for TF-IDF)
#Training an SVM classifier
svm_model = SVC(kernel="linear", C=1.0, probability=True)
svm_model.fit(X_train_tfidf, y_train)

# Makimg predictions
y_pred = svm_model.predict(X_test_tfidf)

# Evaluating the  Model Performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")
print("Classification Report:\n", classification_report(y_test, y_pred))

# Installing dependencies (run only once)
#!pip install gensim scikit-learn pandas

# Importing required libraries
import pandas as pd
#import numpy as np
import gensim.downloader as api
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report

# Loading the dataset
# Make sure 'news lie.csv' is uploaded to your Colab session (left sidebar > Files)
df = pd.read_csv("path.csv")

# Loading pre-trained Word2Vec model (first time takes a few minutes)
w2v_model = api.load("word2vec-google-news-300")

# Function to convert sentence to averaged Word2Vec vector
def sentence_vector(sentence):
    words = sentence.lower().split()
    valid_words = [w for w in words if w in w2v_model]
    if not valid_words:
        return np.zeros(300)
    return np.mean([w2v_model[w] for w in valid_words], axis=0)

#  Vectorizing all statements
X = np.array([sentence_vector(text) for text in df['statement']])
y = df['veracity'].values

from imblearn.over_sampling import RandomOverSampler

# Balancing the data before splitting
ros = RandomOverSampler(random_state=42)
X_balanced, y_balanced = ros.fit_resample(X, y)

# Spliting the balanced data
X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=42)

#  Split into training and testing sets
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardizing vectors for SVM
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#Train SVM classifier
#svm_clf = SVC(kernel='rbf', C=1, gamma='scale')
#svm_clf.fit(X_train_scaled, y_train)

# Hyperparameter Tuning using GridSearchCV
from sklearn.model_selection import GridSearchCV

param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [1, 0.1, 0.01, 0.001],
    'kernel': ['rbf']
}

grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=2, cv=5)
grid.fit(X_train_scaled, y_train)

print(f"Best Parameters: {grid.best_params_}")

# Use the best model for predictions
best_svm = grid.best_estimator_
y_pred = best_svm.predict(X_test_scaled)
print(classification_report(y_test, y_pred))

#  Evaluating the matrix
#y_pred = svm_clf.predict(X_test_scaled)
print(classification_report(y_test, y_pred))

# Use of Random Forest
import pandas as pd
import numpy as np
import gensim.downloader as api
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Load your dataset
df = pd.read_csv("path.csv")

#Load pre-trained Word2Vec model
w2v_model = api.load("word2vec-google-news-300")

#Function to convert sentence to Word2Vec vector
def sentence_vector(sentence):
    words = sentence.lower().split()
    valid_words = [w for w in words if w in w2v_model]
    if not valid_words:
        return np.zeros(300)
    return np.mean([w2v_model[w] for w in valid_words], axis=0)

#Vectorize all statements
X = np.array([sentence_vector(text) for text in df['statement']])
y = df['veracity'].values

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Train Random Forest classifier
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train, y_train)

# Evaluate the model
y_pred = rf_clf.predict(X_test)
print(classification_report(y_test, y_pred))

# Use of XGboost & LGbgm
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

def sentence_vector(sentence):
    words = sentence.lower().split()
    valid_words = [w for w in words if w in w2v_model]
    if not valid_words:
        return np.zeros(300)
    return np.mean([w2v_model[w] for w in valid_words], axis=0)

#  Create feature and label arrays
X = np.array([sentence_vector(text) for text in df['statement']])
y = df['veracity'].values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train XGBoost
xgb_clf = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb_clf.fit(X_train, y_train)

# Predict and evaluate
y_pred_xgb = xgb_clf.predict(X_test)
print(" XGBoost Classification Report:")
print(classification_report(y_test, y_pred_xgb))

# Train LightGBM
lgbm_clf = LGBMClassifier(random_state=42)
lgbm_clf.fit(X_train, y_train)

#  Predict and evaluate
y_pred_lgbm = lgbm_clf.predict(X_test)
print("LightGBM Classification Report:")
print(classification_report(y_test, y_pred_lgbm))


##Audio Deception
#Done in colab

##Video Deception
torch.cuda.is_available()