from torch import cuda, optim
from torch import argmax, manual_seed
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from data.private import file_path
from BERT import ApplicationReviewModel
from Baseline import Baseline

from util import  preprocess_data, model_accuracy, get_dataloader, precision_recall

# Load and preprocess data
questions, traits  = preprocess_data("/home/etowner/Final_project/data/NLP MAlt scores.xlsx")

# Split the data into training and testing sets
train_questions, train_traits = questions[24:], traits[24:]
test_questions, test_traits = questions[:24], traits[:24]

train_loader = get_dataloader(train_questions, train_traits)
test_loader = get_dataloader(test_questions, test_traits)

epochs = 5
learning_rate = 1e-4 


def train_model(model: ApplicationReviewModel, train_loader: DataLoader, 
                dev_loader: DataLoader, epochs: int, learning_rate: float):
    
    device = "cuda" if cuda.is_available() else "cpu"
    criterion = nn.NLLLoss() 
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        total_loss = 0.0
        correct = 0
        total = 0

        model.train()
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            trait_scores = batch[2].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
           
            # compute loss, backward pas, and update weights
            loss = criterion(outputs, trait_scores.argmax(dim=1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            predicted = argmax(outputs, dim=1)
            actual = argmax(trait_scores, dim=1)
            correct += (predicted == actual).sum().item()
            total += trait_scores.size(0)

       
        avg_loss = total_loss / len(train_loader)
        train_accuracy = correct / total
        dev_accuracy = model_accuracy(model, dev_loader, device)
        precision, recall = precision_recall(model, dev_loader, device, num_classes=5)

        print("Epoch", epoch + 1)
        print("Train loss", avg_loss)
        print("Train Accuracy", train_accuracy)
        print("Dev accuracy", dev_accuracy)
        
# Set the device to GPU if available, else CPU
device = "cuda" if cuda.is_available() else "cpu"

# seed the model before initializing weights so that your code is deterministic
manual_seed(457)

model = ApplicationReviewModel().to(device)

#run the baseline model
print("Baseline")
base = Baseline()
base.predict()

# run the BERT model
print("BERT Model")
train_model(model, train_loader, test_loader, epochs, learning_rate)