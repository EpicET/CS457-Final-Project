import random
import numpy as np
import pandas as pd
from util import  preprocess_data
from data.private import file_path
from sklearn.metrics import precision_score, recall_score

# Load and preprocess data
train_questions, train_traits  = preprocess_data(file_path, "train_data")
test_questions, test_traits = preprocess_data(file_path, "test_data")

# Set random seed for reproducibility
random.seed(457)  

class Baseline:
    def __init__(self, train_traits, test_traits):
        self.train_traits = train_traits
        self.test_traits = test_traits
    
    def train(self):
        pass

    def prefict(self):
        traits = ['work', 'teachability', 'commitment', 'Flexibility', 'adventerous']
        
        for target_trait in traits:
            true_scores = []
            for _, row in self.test_traits.iterrows():
                score = int(row[target_trait])
                true_scores.append(score)

            random_scores = [random.randint(1, 5) for _ in range(len(true_scores))]

            correct = 0
            for true_score, random_score in zip(true_scores, random_scores):
                
                if random_score == true_score:
                    correct += 1
            
            accuracy = round((correct / len(true_scores)),2)
            
            precision = round(precision_score(true_scores, random_scores, average='weighted', zero_division=1),2)
            recall = round(recall_score(true_scores, random_scores, average='weighted', zero_division=1),2)
            
            # Print results for each trait
            print(f"Trait: {target_trait} |", f"accuracy: {accuracy} |" , f"Precision: {precision} |", f"Recall: {recall}")
            print()
