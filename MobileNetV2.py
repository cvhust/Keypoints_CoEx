import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.MobileNetV2 import *

class ChildClass(nn.Module):
    def __init__(self):
        super(ChildClass, self).__init__()
        self.fc1= nn.Linear(1280, 256)
        self.fc3=nn.Linear(256,32)
        
        self.relu=nn.ReLU()
        self.sof=nn.Softmax()

        self.drop5= nn.Dropout2d(0.5)

    def forward(self, x):
        x=self.fc1(x)
        x=self.relu(x)
        x=self.drop5(x)
        x=self.fc3(x)
        x=self.soft(x)
        return x


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        
        mbnet= mobilenet_v2(pretrained = True)        
        mbnet.classifier = ChildClass()
        self.model=mbnet      

    def forward(self, img):
    	#forward HBB through MobiletNetV2
        out=self.model(img)
                            
        return out
