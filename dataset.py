import random 
import numpy as np 
import matplotlib.pyplot as plt 
from torch.utils import data
import json
import os
import cv2


def load_json(path):    
    f = open(path,encoding='utf-8')
    dictting = json.load(f)
    return dictting

class JsonDataset(data.Dataset):
    
    def __init__(self,json_author_path, json_pub_path):
        print ("loading json for prepare training")
        self.train_author_dict = load_json(json_author_path)
        self.train_pub_dict = load_json(json_pub_path)  
        self.train_group_list = []
        for key in self.train_author_dict:
            for key1 in self.train_author_dict[key]:
                if len(self.train_author_dict[key][key1])>10:
                    self.train_group_list.append(self.train_author_dict[key][key1])

    def __len__(self):
        return len(self.train_group_list)
    
    def __getitem__(self,index):
        print (self.train_author_dict.keys()[0])
        i = index%len(list(self.train_author_dict.keys()))
        key = list(self.train_author_dict.keys())[index]
        authors = self.train_author_dict[key]
        print (authors)


class PngDataset(data.Dataset):
    
    def __init__(self,png_path):
        self.png_path = png_path
        self.image_list = os.listdir(png_path)
        # random.shuffle(self.image_list)
        self.final_group = int(sorted(self.image_list)[-1].split('_')[0])
        self.start_group = int(sorted(self.image_list)[0].split('_')[0])
        print (self.final_group,self.start_group)
        self.select_num=6

    def __len__(self):
        return (self.final_group)

    def shuffle():
        random.shuffle(self.image_list)
    
    def __getitem__(self,index):
        group_folder_path = os.path.join(self.png_path,self.image_list[index])
        # print (group_folder_path)
        # print (os.listdir(group_folder_path))
        group_list = [os.path.join(group_folder_path,i) for i in os.listdir(group_folder_path)]
        random.shuffle(group_list)
        top_group_list = group_list[:self.select_num]
        image_list =[cv2.imread(file) for file in top_group_list]
        image_array = np.stack(image_list,axis=0)
        image_array = image_array.transpose(0,3,1,2)
        label_array = np.array([index]*self.select_num)
        return image_array,label_array



if __name__ == "__main__":
    
    # new_dataset = JsonDataset("./train/train_author.json","./train/train_pub.json")
    # print (new_dataset[10])
    # print (len(new_dataset))
    new_dataset = PngDataset("./train_jpg_1")
    image_array = new_dataset[11]
    print (image_array.shape)
    



        
