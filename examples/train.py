import _init_path
from models.conv import GatedConv

# model = GatedConv.load("../pretrained/gated-conv.pth")
model = GatedConv('../labels.json')

model.to_train()

model.fit("../data/dataset/annotation/aishell.txt", "../data/dataset/annotation/aishell.txt")
