import torch
import torch.nn as nn
import torchvision.models as models
from collections import OrderedDict
from torchinfo import summary

# net = models.resnet50()
# print(net.fc)

# classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(2048, 128)),
#                                         ('relu1', nn.ReLU()),
#                                         ('dropout1', nn.Dropout(0.5)),
#                                         ('fc2', nn.Linear(128, 10)),
#                                         ('output', nn.Softmax(dim=1))
#                                         ]))

# net.fc = classifier
# print(net.fc)


class Model(nn.Module):
    def __init__(self, net):
        super(Model, self).__init__()
        self.net = net
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc_add = nn.Linear(1001, 10, bias=True)
        self.output = nn.Softmax(dim=1)

    def forward(self, x, add_variable):
        x = self.net(x)
        x = torch.cat((self.dropout(self.relu(x)),
                      add_variable.unsqueeze(1)), 1)
        x = self.fc_add(x)
        x = self.output(x)
        return x


net = models.resnet50()
model = Model(net).to("mps")

summary(net)
