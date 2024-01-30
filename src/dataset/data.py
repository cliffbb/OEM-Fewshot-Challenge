import argparse
import cv2
import numpy as np
from multiprocessing import Pool
from typing import List

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from .classes import get_classes_split
from .utils import make_dataset, Compose, Resize, ToTensor, Normalize


def get_val_loader(args: argparse.Namespace) -> torch.utils.data.DataLoader:
    """
        Build the validation loader.
    """
    val_transform = Compose([Resize(args.image_size),
                             ToTensor(),
                             Normalize(mean=args.mean, std=args.std)])
    
    # ============= Get base and novel classes ===============
    print(f'Data: {args.data_name}')
    classes_split = get_classes_split()
    base_class_list = classes_split['base']
    novel_class_list = classes_split['val']
    
    print('Base classes: ', base_class_list, 'Novel classes: ', novel_class_list)
    args.num_classes_tr = len(base_class_list) + 1  # +1 for bg
    args.num_classes_val = len(novel_class_list)
    
    # ===================== Build loader =====================
    val_sampler = None
    val_data = MultiClassValData(transform=val_transform,
                                 base_class_list=base_class_list,
                                 novel_class_list=novel_class_list,
                                 args=args)
    
    val_loader = torch.utils.data.DataLoader(val_data,
                                             batch_size=args.batch_size_val,
                                             drop_last=False,
                                             shuffle=args.shuffle_test_data,
                                             num_workers=args.workers,
                                             pin_memory=args.pin_memory,
                                             sampler=val_sampler)
    return val_loader


def get_image_and_label(image_path, label_path):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = np.float32(image)
    label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
    if image.shape[0] != label.shape[0] or image.shape[1] != label.shape[1]:
        raise (RuntimeError('Image & label shape mismatch: ' + image_path + ' ' + label_path + '\n'))
    return image, label


class MultiClassValData(Dataset):
    def __init__(self, transform: transforms.Compose, base_class_list: List[int], 
                 novel_class_list: List[int], args: argparse.Namespace):               
        self.transform = transform
        self.base_class_list = base_class_list  # Does not contain bg
        self.novel_class_list = novel_class_list   # Does not contain bg
        self.shot = args.shot
        assert self.shot == 5, 'only 5-shot is allowed!'

        # Get query set list
        self.query_data_list = make_dataset(args.data_root, args.images_dir, args.labels_dir, args.data_list_path,
                                            self.base_class_list + self.novel_class_list, is_support_set=False)
        # Get support set list
        self.support_data_list = make_dataset(args.data_root, args.images_dir, args.labels_dir, 
                                              args.data_list_path, self.novel_class_list, is_support_set=True) 
     
    @property
    def num_novel_classes(self):
        return len(self.novel_class_list)

    @property
    def all_classes(self):
        return [0] + self.base_class_list + self.novel_class_list
        
    def __len__(self):
        return len(self.query_data_list)

    def __getitem__(self, index):  # It only gives the query set 
        image_path, label_path = self.query_data_list[index]
        qry_img, label = get_image_and_label(image_path, label_path)
        qry_img_name = image_path.split("/")[-1].split(".")[0]
             
        if self.transform is not None:
            qry_img, label = self.transform(qry_img, label)
        valid_pixels = (label != 255).float()
        
        return qry_img, label, valid_pixels, qry_img_name

    def get_support_set(self):   # It gives support set
        image_list, label_list = list(), list()

        for c in self.novel_class_list:
            class_data_list = self.support_data_list[c]
            num_class_list = len(class_data_list)
            assert num_class_list == self.shot
            
            found_images_count = 0
            for idx in range(num_class_list):
                image_path, label_path = class_data_list[idx]
                image, label = get_image_and_label(image_path, label_path)
                image_list.append(image)
                label_list.append(label)
                found_images_count += 1
                
            print(f'Number of support images for novel class {c} :', found_images_count)

        print('Total number of support images (support set):', len(self.support_data_list) * self.shot)
        print('Total number of query images (query set):', len(self.query_data_list))
        
        transformed_image_list, transformed_label_list = list(), list()
        with Pool(self.shot) as pool:
            for transformed_i, transformed_l in pool.starmap(self.transform, zip(image_list, label_list)):
                transformed_image_list.append(transformed_i.unsqueeze(0))
                transformed_label_list.append(transformed_l.unsqueeze(0))
            pool.close()
            pool.join()

        spprt_imgs = torch.cat(transformed_image_list, 0)
        spprt_labels = torch.cat(transformed_label_list, 0)
        
        return spprt_imgs, spprt_labels
 