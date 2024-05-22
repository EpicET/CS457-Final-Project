import random
import numpy as np
import pandas as pd
from util import  preprocess_data
from data.private import file_path
from sklearn.metrics import precision_score, recall_score

# Load and preprocess data
questions, traits  = preprocess_data(file_path)

# Split the data into training and testing sets
train_questions, train_traits = questions[24:], traits[24:]
test_questions, test_traits = questions[:24], traits[:24]

# Set random seed for reproducibility
random.seed(457)  

class Baseline:
    def __init__(self):
        # Initialize the class with training and testing traits
        self.train_traits = train_traits
        self.test_traits = test_traits
        self.labels = ['work', 'teachability', 'commitment', 'flexibility', 'adventerous']
    
   
    def predict(self):
         # Loop over each label
        for target_trait in self.labels:
            true_scores = []
            for _, row in self.test_traits.iterrows():
                 # Get the score for the current trait
                score = int(row[target_trait])
                true_scores.append(score)

            # Generate random scores
            random_scores = [random.randint(1, 5) for _ in range(len(true_scores))]

            correct = 0
            for true_score, random_score in zip(true_scores, random_scores):
                if random_score == true_score:
                    correct += 1
            
            # Calculate accuracy, precision, and recall
            accuracy = round((correct / len(true_scores)), 2)
            precision = round(precision_score(true_scores, random_scores, average='weighted', zero_division=1), 2)
            recall = round(recall_score(true_scores, random_scores, average='weighted', zero_division=1), 2)
            
            # Print results for each trait
            print(f"Trait: {target_trait} |", f"Test Accuracy: {accuracy} |", f"Precision: {precision} |", f"Recall: {recall}")
            print()
