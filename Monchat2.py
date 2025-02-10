from matplotlib import pyplot as plt
import numpy as np
import pandas as pd  
import os
import re
from sklearn.model_selection import learning_curve, train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import nltk

# Download NLTK tokenizer
nltk.download('punkt')

# Load the dataset
file_path = 'dataset.csv'
data = pd.read_csv(file_path, on_bad_lines='skip', quotechar='"', delimiter=',')

# Display first few rows before cleaning
print("Raw Data Sample:")
print(data.head())

# **1. Remove Duplicates & Missing Values**
data.drop_duplicates(inplace=True)
data.dropna(inplace=True)

# **2. Clean Text Data**
def clean_text(text):
    text = str(text).lower()  # Convert to lowercase
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove special characters
    return text.strip()

# Apply cleaning function to the text column
data.iloc[:, 1] = data.iloc[:, 1].apply(clean_text)

# **3. Validate Labels**
data = data[data.iloc[:, 2].notnull()]  # Remove rows with missing labels

# **4. Save the cleaned dataset**
cleaned_file_path = 'data/cleaned_dataset.csv'
os.makedirs('data', exist_ok=True)  # Create directory if it doesn't exist
data.to_csv(cleaned_file_path, index=False) 

print("Cleaned dataset saved.")

# **5. Use Cleaned Data for Model Training**
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data.iloc[:, 1])  # Vectorize cleaned text
y = data.iloc[:, 2]  # Target labels

# Ensure `y` has more than one unique class
if len(np.unique(y)) < 2:
    raise ValueError("Target variable `y` must have at least two unique classes.")

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Save training & testing data
train_data = pd.DataFrame(X_train.toarray(), columns=vectorizer.get_feature_names_out())
train_data['label'] = y_train
train_data.to_csv('data/training_dataset.csv', index=False)

test_data = pd.DataFrame(X_test.toarray(), columns=vectorizer.get_feature_names_out())
test_data['label'] = y_test
test_data.to_csv('data/testing_dataset.csv', index=False)

print("Training and testing datasets created successfully.")

# **6. Train the Naive Bayes classifier**
model = MultinomialNB()
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, zero_division=1)

# **7. Learning Curve**
train_sizes, train_scores, test_scores = learning_curve(
    model, X, y, cv=2, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10)
)



train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

plt.figure()
plt.plot(train_sizes, train_mean, 'o-', color='r', label='Training score')
plt.plot(train_sizes, test_mean, 'o-', color='g', label='Cross-validation score')

plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='r')
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color='g')

plt.title('Learning Curve')
plt.xlabel('Training Examples')
plt.ylabel('Score')
plt.legend(loc='best')
plt.grid()
plt.show()

print(f"Accuracy: {accuracy}")
print("Classification Report:")
print(report)
