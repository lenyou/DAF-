
import os
import pandas as pd 
import json 
import cv2
import numpy as np 
import matplotlib.pyplot as plt
import random
import time



def load_json(path):    
    f = open(path,encoding='utf-8')
    dictting = json.load(f)
    return dictting


def parint_large_string(canvas,string,coor,font=cv2.FONT_HERSHEY_SIMPLEX,size=0.5,value=0,paint_size=1,enter_interval=80):
    string_length = len(string)
    string_rows = int(string_length/enter_interval)+2
    last_row = coor[1]
    for i in range(string_rows):
        last_row = coor[1]+14*i
        if i == (string_rows-1):
            canvas = cv2.putText(canvas, string[(string_rows-1)*enter_interval:], (coor[0],last_row), font, 0.5, 0, 1)
        canvas = cv2.putText(canvas, string[i*enter_interval:(i+1)*enter_interval], (coor[0],last_row), font, 0.5, 0, 1)
    last_row += 1
    return canvas,last_row
            

def generate_Pic(info):
    tic = time.time()
    canvas = np.ones((768,768))
    font = cv2.FONT_HERSHEY_SIMPLEX
    start_row = 14
    if "authors" in info:
        authors = info["authors"]
        random.shuffle(authors)
        
        # write aut
        for author_id,author in enumerate(authors):
            if "org"  in author:
                author_string = "NAME: %s"%author['name'] + " org: %s"%author['org']
            else:
                author_string = "NAME: %s"%author['name']
            canvas,last_row = parint_large_string(canvas,author_string,(2,start_row))
            start_row = last_row
    if "title" in info:
        title_string = "TITLE: %s"%info["title"]
        canvas,last_row = parint_large_string(canvas,title_string,(2,start_row))
        start_row = last_row
    if "abstract" in info:
        abstract = "ABSTRACT: %s"%info["abstract"]
        canvas,last_row = parint_large_string(canvas,abstract,(2,start_row))
        start_row = last_row
    if "year" in info:
        year = "YEAR: %s"%info["year"]
        canvas,last_row = parint_large_string(canvas,year,(2,start_row))
        start_row = last_row
    if "keywords" in info:
        keywords = "KEYWORDS: %s"%info["keywords"]
        canvas,last_row = parint_large_string(canvas,keywords,(2,start_row))
        start_row = last_row
    toc = time.time()
    return canvas.astype("uint8")

if __name__ == "__main__":
    target_path = "./val_jpg_1"
    if not os.path.exists(target_path):
        os.mkdir(target_path)
    train_author_path = "/media/liu/large_disk/advanced_machine_learning/sna_data/sna_valid_author_raw.json"
    train_pub_path = "/media/liu/large_disk/advanced_machine_learning/sna_data/sna_valid_pub.json"
    train_author_dict = load_json(train_author_path)
    train_pub_dict = load_json(train_pub_path)
    author_id = 0
    for key in train_author_dict:
        if len(train_author_dict[key])<10:
            continue
        author_id+=1
        folder_path = os.path.join(target_path,"%07d"%author_id)
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)
        for paper_key in train_author_dict[key]:
            values = train_pub_dict[paper_key]
            canvas = generate_Pic(values)
            cv2.imwrite(os.path.join(folder_path,"%07d_%s.jpg"%(author_id,paper_key)),canvas*255)
    print (author_id)
    