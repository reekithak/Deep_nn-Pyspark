import sqlite3
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix
from torch.utils.data import Dataset, DataLoader
from models import *
import torch
from sklearn.decomposition import PCA
from tqdm import tqdm
import sys
from matplotlib import pyplot as plt

connection = sqlite3.connect("./DB/drugs.db")
print("Loading Data...")
df = pd.read_sql_query("SELECT * from train_set0", connection)
# print(df.head())
print(df.shape)
# df["target"].value_counts()[1000:1100].plot(kind="bar")
unique_pbdid = set(",".join(df["target"].values.flatten().tolist()).split(","))
pbdid_dict = {v:k for k, v in dict(enumerate(unique_pbdid)).items()}
del unique_pbdid
print(len(pbdid_dict))
print("Data Loaded...")
df = df.sample(frac=1).reset_index(drop=True)
df.isna().sum()

x = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
# del df
scaler = StandardScaler()
scaler.fit(x[:, 1:])

x_t, x_v, y_t, y_v = train_test_split(x, y, test_size=0.01, random_state=42)
x_t[:, 1:] = scaler.transform(x_t[:, 1:])
x_v[:, 1:] = scaler.transform(x_v[:, 1:])
del x
del y
print("---Features Scaled---")

class GraphDataset(Dataset):
    def __init__(self, x, y, pbdid_dict):
        self.cids = df["cid"].unique()
        self.x = x
        self.y = y
        self.pbdid_dict = pbdid_dict

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        cid = self.cids[idx]
        index = np.where(self.x[:, 0] == cid)[0]
        x = self.x[index, 1:].tolist()
        y = np.zeros(len(self.pbdid_dict))
        targets = ",".join(self.y[index]).split(",")
        targets = [self.pbdid_dict[_id] for _id in targets if _id != ""]
        y[targets] = 1
        return torch.tensor(x), torch.tensor(y.tolist())

train_data = GraphDataset(x_t, y_t, pbdid_dict)
val_data = GraphDataset(x_v, y_v, pbdid_dict)
#
train_loader = DataLoader(train_data, shuffle=True, batch_size=64)
val_loader = DataLoader(val_data, shuffle=True, batch_size=64)

# x, y = next(iter(train_loader))
# print(x.shape, y.shape)
del df
# del x_t
# del y_t
# del x_v
# del y_v
# x, y = next(iter(train_loader))
# print(x, y)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = LinearClassifier(len(pbdid_dict)).to(device)
# criterion = torch.nn.NLLLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

train=False
if train:
    if len(sys.argv) > 1:
        EPOCHS = int(sys.argv[1])
    else:
        EPOCHS = 10

    print("epoch\t train_loss\t val_loss")
    for epoch in range(1, EPOCHS+1):
        train_loss = 0.0
        val_loss = 0.0
        train_loop = tqdm(train_loader, total=len(train_loader), leave=False)
        val_loop = tqdm(val_loader, total=len(val_loader), leave=False)
        model.train()
        for batch in train_loop:
            x, y = batch
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            yhat, loss = model(x.float(), y)
            # loss = criterion(yhat, y)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()

            train_loop.set_description(f"Epoch [{epoch}/{EPOCHS}]")
            train_loop.set_postfix(loss=loss.item())

        model.eval()
        with torch.no_grad():
            for batch in val_loop:
                x, y = batch
                x, y = x.to(device), y.to(device)
                yhat, loss = model(x.float(), y)
                # loss = criterion(yhat, y)
                val_loss += loss.item()
        print(f"{epoch}\t {round(train_loss/len(train_loader), 4)}\t {round(val_loss/len(val_loader), 4)}")
        torch.save(model.state_dict(), f"./models/lc-stage-2-{epoch}.pt")
valid = False
if valid:
    model.load_state_dict(torch.load("./models/lc-stage-2-10.pt", map_location=torch.device("cpu")))
    val_loop = tqdm(val_loader, total=len(val_loader), leave=False)
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for batch in val_loop:
            x, y = batch
            x, y = x.to(device), y.to(device)
            yhat, _ = model(x.float(), y)
            yhat = yhat.argmax(1)
            predictions = yhat == y
            correct += sum(predictions.numpy().astype(int))
            total += y.shape[0]
    print("Model Accuracy", round(correct/total, 4) * 100.0, "%")
