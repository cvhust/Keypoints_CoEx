import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.MobileNetV2 import *

class ChildClass(nn.Module):
    def __init__(self):
        super(ChildClass, self).__init__()
        self.fc1= nn.Linear(1280, 256)
        self.relu=nn.ReLU()
        self.drop5= nn.Dropout2d(0.5)
        self.fc3=nn.Linear(256,32)
    def forward(self, x):
        x=self.fc1(x)
        x=self.relu(x)
        x=self.drop5(x)
        x=self.fc3(x)
        x=self.relu(x)
        return x


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()

        self.ln3 = nn.Linear(86, 64)
        self.ln4 = nn.Linear(64, 64)
        self.ln6 = nn.Linear(64, 64)
        self.ln8 = nn.Linear(96,32)

        self.relu = nn.ReLU()
        self.soft = nn.Softmax()

        self.dropout2=nn.Dropout2d(0.2)
        self.dropout3=nn.Dropout2d(0.3)
        
        mbnet= mobilenet_v2(pretrained = True)        
        mbnet.classifier = ChildClass()
        self.model=mbnet      

    def forward(self,kp,img):
    	#forward HBB through MobiletNetV2
        img=self.model(img)
        
        #forward keypoints through FC
        kp=self.ln3(kp)
        kp=self.relu(kp)
        kp=self.dropout2(kp)
        kp=self.ln4(kp)
        kp=self.relu(kp)
        kp=self.dropout3(kp)
        kp=self.ln6(kp)
        kp=self.relu(kp)
	
        #concatenation
        out= torch.cat([kp, img], dim=1)
        
        out=self.ln8(out)
        out=self.soft(out)
                            
        return out
