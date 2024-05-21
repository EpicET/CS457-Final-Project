from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import precision_score, recall_score
import matplotlib.pyplot as plt
from data.private import file_path
from util import  preprocess_data
import pandas as pd
import numpy as np

# Load and preprocess data
questions, traits  = preprocess_data(file_path)
train_questions, train_traits = questions[24:], traits[24:]
test_questions, test_traits = questions[:98], traits[:98]


# Vectorize the text data
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(train_questions)
X_test = vectorizer.transform(test_questions)

train_accuracies = []
test_accuracies = []
precisions = []
recalls = []

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

    # Append results to lists
    train_accuracies.append(train_accuracy)
    test_accuracies.append(test_accuracy)
    precisions.append(precision)
    recalls.append(recall)

    # Print results for each trait
    print(f"Trait: {target_trait} |", f"train accuracy: {train_accuracy} |", f"Test accuracy: {test_accuracy} |", f"Precision: {precision} |", f"Recall: {recall}")
    print()


scores = pd.DataFrame(columns=['train_accuracy', 'test_accuracy', 'precision', 'recall'])
scores['train_accuracy'] = train_accuracies
scores['test_accuracy'] = test_accuracies
scores['precision'] = precisions
scores['recall'] = recalls
scores.index = traits

fig, ax = plt.subplots()

# Create the table
table = ax.table(cellText=scores.values, colLabels=scores.columns, rowLabels=scores.index, loc='center', cellLoc='center')

# Modify table properties
table.auto_set_font_size(True)
# table.set_fontsize(12)
# table.scale(1.2, 1.2)  # Adjust scaling for better readability

# Hide axes
ax.axis('off')

plt.title("Naive Bayes Evaluation")

# Show the table
plt.show()