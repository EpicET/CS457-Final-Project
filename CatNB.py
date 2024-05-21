from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import precision_score, recall_score
import matplotlib.pyplot as plt
from data.private import file_path
from util import preprocess_data
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

# Load and preprocess data
train_questions, train_traits  = preprocess_data(file_path, "train_data")
test_questions, test_traits = preprocess_data(file_path, "test_data")

# Combine all questions for encoding
all_questions = pd.concat([train_questions, test_questions])
all_traits = pd.concat([train_traits, test_traits])

# Ordinal encode the text data
encoder = OrdinalEncoder()
# all_questions = np.array(all_questions).reshape(-1, 1) # Reshape to a 2D array

encoded_questions = encoder.fit_transform(all_questions)



train_accuracies = []
test_accuracies = []
precisions = []
recalls = []

# Iterate over each trait
traits = ['work', 'teachability', 'commitment', 'Flexibility', 'adventurous']
for target_trait in traits:
    y = all_traits[target_trait].values
    y = np.round(y).astype(int)
    
    # Split the data into 20% train and 80% test sets
    X_train, X_test, y_train, y_test = train_test_split(
        encoded_questions, 
        y, 
        test_size=0.8, 
        random_state=42
    )

    # Initialize and train the Categorical Naive Bayes model
    model = CategoricalNB()
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
    print(f"Trait: {target_trait} | train accuracy: {train_accuracy} | Test accuracy: {test_accuracy} | Precision: {precision} | Recall: {recall}")
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

# Hide axes
ax.axis('off')

plt.title("Naive Bayes Evaluation")

# Show the table
plt.show()
