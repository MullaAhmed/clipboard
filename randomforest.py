import pandas as pd
# Import necessary libraries
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC


df=pd.read_csv("final.csv")
df.head(2)

texts=df["Sentences"].values
labels=df["Labels"].values


# Convert labels to numerical values (0, 1, 2, 3)
label_mapping = {label: idx for idx, label in enumerate(set(labels))}
numeric_labels = [label_mapping[label] for label in labels]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(texts, numeric_labels, test_size=0.2, random_state=42,stratify=numeric_labels)

# Create a CountVectorizer to convert text data into a bag-of-words representation
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
logistic_regression = LogisticRegression(max_iter=1000, random_state=42)
naive_bayes = MultinomialNB()
svm_classifier = SVC(kernel='linear', probability=True, random_state=42)  # You can adjust the kernel and other parameters

# Create a StackingClassifier with Random Forest, Logistic Regression, Naive Bayes, and SVM
stacking_classifier = StackingClassifier(
    estimators=[
        ('random_forest', rf_classifier),
        ('logistic_regression', logistic_regression),
        ('naive_bayes', naive_bayes),
        ('svm', svm_classifier)
    ],
    final_estimator=LogisticRegression(),
    stack_method='auto',
    n_jobs=-1
)

# Train the stacking classifier
stacking_classifier.fit(X_train_vectorized, y_train)


model=stacking_classifier
# Make predictions on the test set
predictions = model.predict(X_test_vectorized)

# Evaluate the model
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy:.2f}")
# Display classification report
print("Classification Report:")
print(classification_report(y_test, predictions, target_names=label_mapping.keys()))

import pickle
model_filename = "text_classification_model.pkl"
with open(model_filename, 'wb') as model_file:
    pickle.dump(stacking_classifier, model_file)

print(f"Model saved to {model_filename}")

# Load the model back using pickle
with open(model_filename, 'rb') as model_file:
    model = pickle.load(model_file)