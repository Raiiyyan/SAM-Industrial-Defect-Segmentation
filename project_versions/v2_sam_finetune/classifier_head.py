import torch
import torch.nn as nn

CLASS_VOCAB = {0: "Flawless", 1: "Surface Scratch", 2: "Structural Crack", 3: "Hole / Puncture", 4: "Inclusion", 5: "Missing Component", 6: "Discoloration / Stain", 7: "Geometric Deformation"}
NUM_CLASSES = 8

class DefectClassifierHead(nn.Module):
    def __init__(self, input_dim=256, num_classes=8, dropout=0.3):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)