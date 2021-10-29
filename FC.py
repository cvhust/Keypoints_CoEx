import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()

        self.sig = nn.Sigmoid()
        self.soft=nn.Softmax()

        self.dropout2=nn.Dropout2d(0.2)
        self.dropout3=nn.Dropout2d(0.3)

        self.ln4 = nn.Linear(86, 128)
        self.ln5 = nn.Linear(128, 64)
        self.ln6 = nn.Linear(64, 32)
        self.ln8 = nn.Linear(32,15)
        

    def forward(self, kp):

        kp = self.ln4(kp)
        kp = self.sig(kp)
        kp = self.dropout2(kp)
        kp = self.ln5(kp)
        kp = self.sig(kp)
        kp = self.dropout3(kp)
        kp = self.ln6(kp)
        kp = self.sig(kp)
        kp = self.dropout2(kp)
        kp = self.ln8(kp)

        return kp
