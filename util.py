import pandas as pd
from BERT import ApplicationReviewModel
import torch
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, TensorDataset

def preprocess_data(data_path: str):
    # Load the data
    data = pd.read_excel(data_path)

    # Drop rows with NaN values
    data = data.dropna()

    # Separate questions and traits
    # Convert questions into a single text column
    questions = data[['Q1', 'Q2', 'Q3']].fillna('').agg(' '.join, axis=1)
    traits = data[['work', 'teachability', 'commitment', 'flexibility', 'adventerous']]
    
    return questions, traits

def get_dataloader(questions: pd.Series, traits: pd.DataFrame, batch_size: int=4):
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    tokenized = tokenizer(questions.tolist(), padding=True, truncation=True, return_tensors="pt")
    
    input_ids = tokenized['input_ids']
    attention_masks = tokenized['attention_mask']
    trait_scores = torch.tensor(traits.values, dtype=torch.long)

    dataset = TensorDataset(input_ids, attention_masks, trait_scores)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    return dataloader

def model_accuracy(model: ApplicationReviewModel, dataloader: DataLoader, device):
    """Compute the accuracy of a regression model

    Args:
        model (ApplicationReviewModel): a regression model
        dataloader (DataLoader): a pytorch data loader to test the model with
        device (string): cpu or cuda, the device that the model is on

    Returns:
        float: the accuracy of the model
    """
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for batch in dataloader:
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            trait_scores = batch[2].to(device)

            outputs = model(input_ids, attention_mask)
            
            predicted = torch.argmax(outputs, dim=1)
            actual = torch.argmax(trait_scores, dim=1)
            correct += (predicted == actual).sum().item()
            total += trait_scores.size(0)
        acc = correct / total
        return acc

def precision_recall(model: ApplicationReviewModel, dataloader: DataLoader, device, num_classes):
    """Compute precision and recall of a classification model for each class

    Args:
        model (ApplicationReviewModel): a classification model
        dataloader (DataLoader): a pytorch data loader to test the model with
        device (string): cpu or cuda, the device that the model is on
        num_classes (int): number of classes in the classification problem

    Returns:
        tuple: precision and recall for each class
    """
    model.eval()
    with torch.no_grad():
        true_positives = torch.zeros(num_classes).to(device)
        false_positives = torch.zeros(num_classes).to(device)
        false_negatives = torch.zeros(num_classes).to(device)
        
        for batch in dataloader:
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            trait_scores = batch[2].to(device)

            outputs = model(input_ids, attention_mask)
            predicted = torch.argmax(outputs, dim=1)
            actual = torch.argmax(trait_scores, dim=1)
            
            for i in range(num_classes):
                true_positives[i] += ((predicted == i) & (actual == i)).sum().item()
                false_positives[i] += ((predicted == i) & (actual != i)).sum().item()
                false_negatives[i] += ((predicted != i) & (actual == i)).sum().item()

        precision = true_positives / (true_positives + false_positives + 1e-10)  # Add a small value to avoid division by zero
        recall = true_positives / (true_positives + false_negatives + 1e-10)  # Add a small value to avoid division by zero

        return precision, recall



