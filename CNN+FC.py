import torch
import torch.nn as nn
import torch.nn.functional as F

def conv_block(input_size, output_size):
    block = nn.Sequential(
        nn.Conv2d(input_size, output_size, (5, 5)), nn.ReLU(), nn.BatchNorm2d(output_size), nn.MaxPool2d((2, 2)),
    )

    return block


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        
        self.conv1 = conv_block(3, 16)
        self.conv2 = conv_block(16, 32)
        self.conv3 = conv_block(32, 64)

        self.ln3 = nn.Linear(86, 64)
        self.ln4 = nn.Linear(64, 64)
        self.ln6 = nn.Linear(64, 32)
        self.ln8 = nn.Linear(48,32)

        self.relu = nn.ReLU()
        self.soft = nn.Softmax()

        self.dropout = nn.Dropout2d(0.5)
        self.dropout2=nn.Dropout2d(0.2)
        self.dropout3=nn.Dropout2d(0.3)

    def forward(self,kp,img):

        #forward image through CNN
        img = self.conv1(img)
        img= self.dropout2(img)
        img = self.conv2(img)
        img=self.dropout3(img)        
        img = self.conv3(img)
        img=self.dropout(img)   
        img = img.reshape(img.shape[0], -1)
        img = self.ln1(img)
        img=self.relu(img)
        img=self.dropout2(img)

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

        out=self.soft(out)
        out=self.ln8(out)
                            

        return out
