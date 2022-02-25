import os 
import sys 
import warnings
import numpy as np 
import pandas as pd 
from tqdm import tqdm
import seaborn as sns 
import matplotlib.pyplot as plt 

import torch 
from torch_geometric.data import DataLoader
from dataset import MoleculeDataset
from utils import count_parameters

from model import GNN

warnings.filterwarnings("ignore")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device {device}")


root_path = os.path.join(os.getcwd()[:-4],"Data/")
test_file_name = "test_data.csv"
train_file_name = "train_data.csv"

train_dataset = MoleculeDataset(root_path, train_file_name)
test_dataset = MoleculeDataset(root_path, test_file_name)

train_loader = DataLoader(train_dataset, batch_size = 64, shuffle = True)
test_loader = DataLoader(test_dataset, batch_size = 64, shuffle = True)


in_features = 30
hidden_features = 32
edge_dim = 11 

model = GNN(
    in_features=in_features,
    hidden_features=hidden_features,
    out_features=1,
    edge_dim=edge_dim
)

model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.MSELoss()
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)

def train(model, optimizer, criterion):
    running_loss = 0.0
    model.train()
    for mol_graph in train_loader:
        x = mol_graph.x.to(device)
        y = mol_graph.y.to(device)
        edge_attr = mol_graph.edge_attr.to(device)
        edge_index = mol_graph.edge_index.to(device)
        batch = mol_graph.batch.to(device)
        
        optimizer.zero_grad()
        out = model(x, edge_attr, edge_index, batch)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        #scheduler.step()
        running_loss += loss.item()
    return running_loss / len(train_loader)

def test(model, criterion):
    model.eval()
    running_loss = 0.0 
    with torch.no_grad():
        for mol_graph in test_loader:
            x = mol_graph.x.to(device)
            y = mol_graph.y.to(device)
            edge_attr = mol_graph.edge_attr.to(device)
            edge_index = mol_graph.edge_index.to(device)
            batch = mol_graph.batch.to(device)
            
            out = model(x, edge_attr, edge_index, batch)
            loss = criterion(out, y)
            running_loss += loss.item()
    return running_loss / len(test_loader)


train_loss = []
test_loss = []

begin_test_loss = 0.0 
step = 5
count = 0

for epoch in tqdm(range(1, 6)):
    current_train_loss = train(model, optimizer, criterion)
    current_test_loss = test(model, criterion)
    if epoch == 1:
        begin_test_loss = current_test_loss
    
    print(f"Epoch {epoch} Train Loss: {current_train_loss} Test Loss: {current_test_loss}")
    
    train_loss.append(current_train_loss)
    test_loss.append(current_test_loss)

    if current_test_loss > begin_test_loss:
        print("loss not decreased ...")
        count += 1
    else:
        begin_test_loss = current_test_loss
        torch.save(model, "trained_model.pth")
        count = 0
    
    if count >= 3:
        print("Early stopping ...")
        break

plt.figure(figsize=(10, 7))
plt.plot(train_loss)
plt.plot(test_loss)
plt.legend(["Train loss", "Test Loss"])
plt.title("The curve showing the loss curve per epoch for both train and test.")
plt.show()