import torch
from torch.utils.data import Dataset, DataLoader
import cv2

class ImageDataset(Dataset):

    def __init__(self, X_path,Y_path, img_path):
        """X_path: path to file contains keypoints of the gestures (use in Two-pipeline and FC models)
            Y_path: path to file contains the labels of the gestures(images)
            img_path: path to file contains the image paths(use in Two-pipeline and MobileNetV2 models)
        """

        #load and process the keypoints
        x_norm_pre=load_X(X_path)
        self.X_norm = norm_X(x_norm_pre)

        #load the labels
        self.Y_set=load_Y(Y_path)
	
        #load image path
        f=open(img_path,"r")
        img_set=[]
        for t in f:
            #remove character "\n" at the end of each line
            img_set.append(t[:len(t)-1]) 
        self.img_set=np.asarray(img_set)
        f.close()


    def __len__(self):
        return len(self.Y_set)

    def __getitem__(self, idx):

        try:
            image = cv2.imread(self.img_set[idx])
        except:
            pass
        image=cv2.resize(image, (224,224)) # size (128,128) with CNN + FC
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], \
            std=[0.229, 0.224, 0.225])])
        image =transform(image)

        x_kp= torch.from_numpy(self.X_norm[idx]).float()

        y=self.Y_set[idx]

        return x_kp, image, y

image_data_train = ImageDataset('../data/X_train.txt', '../data/Y_train.txt', '../data/img_train_path.txt')
image_data_val = ImageDataset(X_validation_path, Y_validation_path, img_validation_path)
image_data_test = ImageDataset(X_test_path, Y_test_path, img_test_path)

#training dataloader
dataloader_train = DataLoader(dataset=image_data_train, batch_size, shuffle= True, num_workers)

#validation dataloader
dataloader_val = DataLoader(dataset=image_data_val, batch_size, shuffle= False, num_workers)

#test dataloader
dataloader_test = DataLoader(dataset=image_data_test, batch_size, shuffle= False, num_workers)

