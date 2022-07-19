import os
from os import path as osp
import zipfile
import pickle
import shutil
import pathlib
import json
import numpy as np
from tqdm import tqdm

def main():
    train_annot, val_annot = pickle.load(open('clean_train.pkl', 'rb')), pickle.load(open('clean_valid.pkl', 'rb'))
    data_path = 'data_hmor'
    os.makedirs(data_path, exist_ok=True)
    for annot_name, annot in zip(('train', 'valid'), (train_annot, val_annot)):
        for term in tqdm(annot):
            img_path = term['image_name']
            subject, view = img_path.split('/')[-4], img_path.split('/')[-2]
            calibration = json.load(open(osp.join('data', *(img_path.split('/')[:-3]), 'calibration_%s.json'%subject)))
            for cal_term in calibration['cameras']:
                if cal_term['type'] == 'hd' and cal_term['name'] == view:
                    term.update(cal_term)
                    break
            os.makedirs(osp.join(data_path, *(img_path.split('/')[:-1])), exist_ok=True)
            shutil.copyfile(osp.join('data', img_path), osp.join(data_path, img_path))
        pickle.dump(annot, open('data_hmor/%s_cam.pkl'%annot_name, 'wb'))



if __name__ == '__main__':
    main()