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
            # Assuming that the model outputs are directly comparable to trait scores
            correct += ((outputs > 0.5).to(int) == trait_scores).sum().item()
            total += trait_scores.size(0)
        acc = correct / total
        return acc




