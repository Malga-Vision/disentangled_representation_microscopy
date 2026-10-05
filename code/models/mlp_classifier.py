



import torch.nn as nn




# Define a simple neural network model
class Classifier(nn.Module):
    def __init__(self, input_dim, n_classes):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, n_classes)


    def forward(self, x):
        x = nn.functional.leaky_relu(self.fc1(x), negative_slope=0.2)
        x = nn.functional.leaky_relu(self.fc2(x), negative_slope=0.2)
        x = nn.functional.leaky_relu(self.fc3(x), negative_slope=0.2)
        return x




