from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import precision_score, recall_score
# from data.private import file_path
from util import  preprocess_data
from Baseline import Baseline
import numpy as np

# Load and preprocess data
questions, traits  = preprocess_data("/home/etowner/Final_project/data/NLP MAlt scores.xlsx")

# Split the data into training and testing sets
train_questions, train_traits = questions[24:], traits[24:]
test_questions, test_traits = questions[:24], traits[:24]


# Vectorize the text data
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(train_questions)
X_test = vectorizer.transform(test_questions)

# Run baseline model 
print("Baseline")
base = Baseline()
base.predict()


print("Multinomial Naive Bayes")
# Iterate over each trait
traits = ['work', 'teachability', 'commitment', 'flexibility', 'adventerous']
for target_trait in traits:
    y_train = train_traits[target_trait]
    y_train = np.round(y_train).astype(int)

    y_test = test_traits[target_trait]
    y_test = np.round(y_test).astype(int)

    # Initialize and train the Multinomial Naive Bayes model
    model = MultinomialNB()
    model.fit(X_train, y_train)

    # Make predictions
    predictions = model.predict(X_test)

    # Evaluate the model
    train_accuracy = round(model.score(X_train, y_train), 2)
    test_accuracy = round(model.score(X_test, y_test), 2)
    precision = round(precision_score(y_test, predictions, average='weighted', zero_division=1), 2)
    recall = round(recall_score(y_test, predictions, average='weighted', zero_division=1), 2)

    # Print results for each trait
    print(f"Trait: {target_trait} |", f"Train accuracy: {train_accuracy} |", f"Test accuracy: {test_accuracy} |", f"Precision: {precision} |", f"Recall: {recall}")
    print()

