import pandas as pd
from BERT import ApplicationReviewModel
import torch
# from datasets import Dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, TensorDataset

def load_data(data_path: str, sheet_name: str):
    # Load the data
    data = pd.read_excel(data_path, sheet_name=sheet_name)

    # Drop rows with NaN values
    data = data.dropna()

    return data

def preprocess_data(data):
    # Separate questions and traits
    questions = data[data[['Q1', 'Q2', 'Q3']].notnull()].copy()
    traits = data[['work', 'teachability', 'commitment', 'Flexibility', 'adventerous', 'overall score']]

    # Convert questions into a single text column
    questions['combined'] = questions.fillna('').apply(lambda x: ' '.join(map(str, x)), axis=1)

    return questions['combined'], traits

def get_dataloader(data, batch_size: int=4):
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    input_ids = []
    attention_masks = []
    trait_scores = []

    # create batches with input_ids, attention_mask, traits
    for i in range(0, len(data), batch_size):
        batch = data[i:i+batch_size]
        texts = [q1 + ' ' + q2 + ' ' + q3 for q1, q2, q3 in zip(batch['Q1'], batch['Q2'], batch['Q3'])]
        tokenized_batch = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        
        input_ids.append(tokenized_batch['input_ids'])
        attention_masks.append(tokenized_batch['attention_mask'])
        trait_scores.append(torch.tensor(batch[['work', 'teachability', 'commitment', 'Flexibility', 'adventerous', 'overall score']].values, dtype=torch.long))

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    trait_scores = torch.cat(trait_scores, dim=0)

    # import datasets gives an error so have to create a dataset
    dataset = TensorDataset(input_ids, attention_masks, trait_scores)
    dataloader = DataLoader(dataset, batch_size=batch_size)

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




