import random
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score

def baseline(test_traits: pd.DataFrame):
    
    traits = ['work', 'teachability', 'commitment', 'Flexibility', 'adventerous']
    
    for target_trait in traits:
        true_scores = []
        for _, row in test_traits.iterrows():
            true_scores.append(row[target_trait])
        random_scores = [random.randint(1, 5) for _ in range(len(true_scores))]

        correct = 0
        for score in true_scores:
            if random_scores[score] == score:
                correct += 1
        
        accuracy = correct / len(true_scores)
        precision = round(precision_score(true_scores, random_scores, average='weighted', zero_division=1),2)
        recall = round(recall_score(true_scores, random_scores, average='weighted', zero_division=1),2)
        
        # Print results for each trait
        print(f"Trait: {target_trait} |", f"accuracy: {accuracy} |" , f"Precision: {precision} |", f"Recall: {recall}")
        print()