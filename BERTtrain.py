from torch import cuda, optim
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm import tqdm

from BERT import ApplicationReviewModel
from util import load_data, preprocess_data, model_accuracy, get_dataloader


train_data = load_data("/home/etowner/Final_project/data/NLP MAlt scores.xlsx", "train_data")
train_questions, train_traits = preprocess_data(train_data)

test_data = load_data("/home/etowner/Final_project/data/NLP MAlt scores.xlsx", "test_data")
test_questions, test_traits = preprocess_data(test_data)


train_loader = get_dataloader(train_data)
test_loader = get_dataloader(test_data)
epochs = 5
learning_rate = 1e-2
device = "cuda" if cuda.is_available() else "cpu"


def train_model(model: ApplicationReviewModel, train_loader, 
                test_loader, epochs: int, learning_rate: float):
    
    
    criterion = nn.NLLLoss() 
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        total_loss = 0.0
        correct = 0
        total = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            trait_scores = batch[2].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            
            loss = criterion(outputs.squeeze(), trait_scores.squeeze())
            print(loss.shape())   
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Still needs to be adjusted
            preds = (outputs > 0.5).squeeze().to(int) 
            correct += (preds == trait_scores).sum().item()
            total += trait_scores.size(0)

        avg_loss = total_loss / len(train_loader)
        train_accuracy = correct / total
        # dev_accuracy = model_accuracy(model, test_loader, device)

        print("Epoch", epoch + 1)
        print("Train loss", avg_loss)
        print("Train Accuracy", train_accuracy)
        # print("Dev accuracy", dev_accuracy)

    # return dev_accuracy

model = ApplicationReviewModel().to(device)

train_model(model, train_loader, test_loader, epochs, learning_rate)