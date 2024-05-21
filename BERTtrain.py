from torch import cuda, optim
from torch import argmax
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm import tqdm
from data.private import file_path
from BERT import ApplicationReviewModel
from util import load_data, preprocess_data, model_accuracy, get_dataloader


train_data = load_data(file_path, "train_data")
train_questions, train_traits = preprocess_data(train_data)

test_data = load_data(file_path, "test_data")
test_questions, test_traits = preprocess_data(test_data)


train_loader = get_dataloader(train_questions, train_traits, train_data)
test_loader = get_dataloader(test_questions, test_traits, test_data)
epochs = 5
learning_rate = 1e-2



def train_model(model: ApplicationReviewModel, train_loader, 
                dev_loader, epochs: int, learning_rate: float):
    
    device = "cuda" if cuda.is_available() else "cpu"
    criterion = nn.NLLLoss() 
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        total_loss = 0.0
        correct = 0
        total = 0

        # model.train()
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            trait_scores = batch[2].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)

            # print("outputs", outputs)
          
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
        print("Epoch", epoch + 1)
        print("Train loss", avg_loss)
        print("Train Accuracy", train_accuracy)
        print("Dev accuracy", dev_accuracy)

# initialize model and dataloaders
device = "cuda" if cuda.is_available() else "cpu"

    # seed the model before initializing weights so that your code is deterministic
# manual_seed(457)

model = ApplicationReviewModel().to(device)

train_model(model, train_loader, test_loader, epochs, learning_rate)