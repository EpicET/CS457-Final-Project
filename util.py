import pandas as pd
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
