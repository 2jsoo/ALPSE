## Import Library
from import_library import *
import warnings
warnings.filterwarnings(action="ignore")


class YOLODataset(Dataset): ## code reference : https://github.com/aladdinpersson/Machine-Learning-Collection/tree/master/ML/Pytorch/object_detection/YOLO
    def __init__(self, data, label, seg_length, fs=100, S=9, B=2, C=1):
        self.data = data
        self.label = label
        self.S = S
        self.B = B
        self.C = C
        self.fs = fs
        self.seg_length = seg_length
        
    def __len__(self):
        return len(self.label)
    
    def __getitem__(self, idx):
        data_ = self.data[idx]
        boxes = self.label[idx]

        label_matrix = torch.zeros((self.S, self.C + 3 * self.B))
        for box in boxes:
            class_label, x_mid, width = box 
            class_label = int(class_label)

            ### Only detect AF/AFL
            if class_label != 0:
                continue

            x_mid = x_mid / (self.fs * self.seg_length)
            width = width / (self.fs * self.seg_length) 


            i = int(self.S * x_mid) 
            x_cell = self.S * x_mid - i

            width_cell = width * self.S 
            
            if label_matrix[i, self.C] == 0: 
                
                label_matrix[i, self.C] = 1 

                box_coordinates = torch.tensor([x_cell, width_cell]) # Box coordinates

                label_matrix[i, self.C+1:self.C+3] = box_coordinates

                
                label_matrix[i, class_label] = 1 # Set one hot encoding for class_label

        self.x_ = data_
        self.y_data = label_matrix
        
        return self.x_, self.y_data