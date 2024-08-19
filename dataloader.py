from import_library import *


def make_data(data_path, fold, train_db, seg_length):
    with open(f'{data_path}/{train_db.split("_")[0]}/Train/{fold}fold/data{seg_length}s.pkl', 'rb') as f:
        train_data = pickle.load(f)
    with open(f'{data_path}/{train_db.split("_")[0]}/Train/{fold}fold/label{seg_length}s.pkl', 'rb') as f:
        train_label = pickle.load(f)
    with open(f'{data_path}/{train_db.split("_")[0]}/Valid/{fold}fold/data{seg_length}s.pkl', 'rb') as f:
        valid_data = pickle.load(f)
    with open(f'{data_path}/{train_db.split("_")[0]}/Valid/{fold}fold/label{seg_length}s.pkl', 'rb') as f:
        valid_label = pickle.load(f) 

    if len(train_db.split("_")) > 1: # ltafdb + nsrdb
        with open(f'{data_path}/{train_db.split("_")[1]}/Train/{fold}fold/data{seg_length}s.pkl', 'rb') as f:
            add_train_data = pickle.load(f)
        with open(f'{data_path}/{train_db.split("_")[1]}/Train/{fold}fold/label{seg_length}s.pkl', 'rb') as f:
            add_train_label = pickle.load(f)
        with open(f'{data_path}/{train_db.split("_")[1]}/Valid/{fold}fold/data{seg_length}s.pkl', 'rb') as f:
            add_valid_data = pickle.load(f)
        with open(f'{data_path}/{train_db.split("_")[1]}/Valid/{fold}fold/label{seg_length}s.pkl', 'rb') as f:
            add_valid_label = pickle.load(f) 
    
        train_data.update(add_train_data)
        train_label.update(add_train_label)
        valid_data.update(add_valid_data)
        valid_label.update(add_valid_label)

    return train_data, train_label, valid_data, valid_label

def make_test_data(data_path, db, seg_length):
    if db != 'ltafdb_nsrdb':
        with open(f'{data_path}/{db}/Test/data{seg_length}s.pkl', 'rb') as f: 
            test_data = pickle.load(f)
        with open(f'{data_path}/{db}/Test/label{seg_length}s.pkl', 'rb') as f:
            test_label = pickle.load(f)
    else:
        with open(f'{data_path}/ltafdb/Test/data{seg_length}s.pkl', 'rb') as f: 
            test_data = pickle.load(f)
        with open(f'{data_path}/ltafdb/Test/label{seg_length}s.pkl', 'rb') as f:
            test_label = pickle.load(f)
        
        test_data.update(pd.read_pickle(f'{data_path}/nsrdb/Test/data{seg_length}s.pkl'))
        test_label.update(pd.read_pickle(f'{data_path}/nsrdb/Test/label{seg_length}s.pkl'))
    
    return test_data, test_label

class CustomDataset(Dataset): ## code reference : https://github.com/aladdinpersson/Machine-Learning-Collection/tree/master/ML/Pytorch/object_detection/YOLO
    def __init__(self, data, label, seg_length=26.88, fs=100, S=9, B=2, C=1):
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