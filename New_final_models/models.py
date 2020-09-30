import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearClassifier(nn.Module):
    def __init__(self, out_features):
        super(LinearClassifier, self).__init__()
        self.clf = nn.Sequential(
            nn.Linear(32, 1024),
            nn.Dropout(0.5),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 2048),
            nn.Dropout(0.5),
            nn.BatchNorm1d(2048),
            nn.Linear(2048, out_features),
            nn.Sigmoid()
        )

    def forward(self, x, y):
        out = self.clf(x)
        loss = nn.BCEWithLogitsLoss()(out, y)
        return out, loss
