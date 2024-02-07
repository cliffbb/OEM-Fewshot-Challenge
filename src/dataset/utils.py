import cv2
import os
import json
import torch
import numpy as np

from tqdm import tqdm
from collections import defaultdict
from functools import partial
from multiprocessing import Pool
from typing import Dict, List, Tuple, TypeVar


IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']

A = TypeVar("A")
B = TypeVar("B")


def is_image_file(filename: str) -> bool:
    filename_lower = filename.lower()
    return any(filename_lower.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(data_root: str,
                 images_dir: str,
                 labels_dir: str,
                 data_list: str,
                 class_list: List[int],
                 is_support_set: bool=False
                 ) -> Tuple[List[Tuple[str, str]], Dict[int, List[Tuple[str, str]]]]:
    """
        Prepare the dataset (img_path, label_path) for support and query sets. 
        
        input:
            data_root : Path to the data directory
            images_dir: Images folder
            labels_dir: Labels folder
            data_list : Path to the .json file that contain the val/test split of images
            class_list: List of classes to process
            is_support_set: bool 
        returns:
            image_label_list: List of tuple of file path (img_path, label_path)
    """
    if not os.path.isfile(data_list):
        raise (RuntimeError("Image list file do not exist: " + data_list + "\n"))
    
    with open(data_list, "r") as f:
        support_query_list = json.load(f)

    list_read = support_query_list["query_set"]
    image_label_list: List[Tuple[str, str]] = []
    
    if is_support_set:
        list_read = [[fname for fname in support_query_list['support_set'][class_]] for class_ in support_query_list['support_set']]     
        image_label_list: Dict[int, List[Tuple[str, str]]] = defaultdict(list)
                
    print(f"Processing data for {class_list}")
    process_partial = partial(process_image, data_root=data_root, images_dir=images_dir, labels_dir=labels_dir)

    with Pool(os.cpu_count() // 2) as pool:
        if is_support_set:
            for p, process_list in enumerate(pool.map(process_partial, tqdm(list_read))):  # mmap
                image_label_list[class_list[p]].extend(process_list) 
        else:
            for _, process_list in enumerate(pool.map(process_partial, tqdm(list_read))):  # mmap
                image_label_list += process_list
        pool.close()
        pool.join()

    return image_label_list 


def process_image(item: [str, list],
                  data_root: str,
                  images_dir: str,
                  labels_dir: str
                  ) ->  List[Tuple[str, str]]: 
    """
        Reads and parses a filename corresponding to 1 image
        
        input:
            item : An image filename or list of images filenames
            data_root : Path to the data directory
            images_dir: Images folder
            labels_dir: Labels folder
    """
    image_label_list: List[Tuple[str, str]] = []
    
    if isinstance(item, list):  
        for class_file in item:
            image_path = os.path.join(data_root, images_dir, class_file) 
            label_path = os.path.join(data_root, labels_dir, class_file) 
            item_set: Tuple[str, str] = (image_path, label_path)
            image_label_list.append(item_set)
        
        return image_label_list
    
    image_path = os.path.join(data_root, images_dir, item) 
    label_path = os.path.join(data_root, labels_dir, item)   
    item_set: Tuple[str, str] = (image_path, label_path)
    image_label_list.append(item_set)
    
    return image_label_list 



# ==================================================================================================
# Transforms have been borrowed from https://github.com/hszhao/semseg/blob/master/util/transform.py
# ==================================================================================================

class Compose(object):
    def __init__(self, segtransform):
        self.segtransform = segtransform

    def __call__(self, image, label=None):
        if label is None:
            for t in self.segtransform:
                image = t(image, None)
            return image
        else:
            for t in self.segtransform:
                image, label = t(image, label)
            return image, label


class ToTensor(object):
    # Converts numpy.ndarray (H x W x C) to a torch.FloatTensor of shape (C x H x W).
    def __call__(self, image, label):
        if not isinstance(image, np.ndarray):
            raise (RuntimeError("segtransform.ToTensor() only handle np.ndarray"
                                "[eg: data readed by cv2.imread()].\n"))
        if len(image.shape) > 3 or len(image.shape) < 2:
            raise (RuntimeError("segtransform.ToTensor() only handle np.ndarray with 3 dims or 2 dims.\n"))
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=2)

        image = torch.from_numpy(image.transpose((2, 0, 1)))
        if not isinstance(image, torch.FloatTensor):
            image = image.float().div(255)
        if label is not None:
            if not isinstance(label, np.ndarray):
                raise (RuntimeError("segtransform.ToTensor() only handle np.ndarray"
                                    "[eg: data readed by cv2.imread()].\n"))
            if not len(label.shape) == 2:
                raise (RuntimeError("segtransform.ToTensor() only handle np.ndarray labellabel with 2 dims.\n"))
            label = torch.from_numpy(label)
            if not isinstance(label, torch.LongTensor):
                label = label.long()
            return image, label
        else:
            return image


class Normalize(object):
    # Normalize tensor with mean and standard deviation along channel: channel = (channel - mean) / std
    def __init__(self, mean, std=None):
        if std is None:
            assert len(mean) > 0
        else:
            assert len(mean) == len(std)
        self.mean = mean
        self.std = std

    def __call__(self, image, label):
        if self.std is None:
            for t, m in zip(image, self.mean):
                t.sub_(m)
        else:
            for t, m, s in zip(image, self.mean, self.std):
                t.sub_(m).div_(s)
        if label is not None:
            return image, label
        else:
            return image


class Resize(object):
    # Resize the input to the given size, 'size' is a 2-element tuple or list in the order of (h, w).
    def __init__(self, size, padding=None):
        self.size = size
        self.padding = padding

    def __call__(self, image, label):

        def find_new_hw(ori_h, ori_w, test_size):
            if ori_h >= ori_w:
                ratio = test_size * 1.0 / ori_h
                new_h = test_size
                new_w = int(ori_w * ratio)
            elif ori_w > ori_h:
                ratio = test_size * 1.0 / ori_w
                new_h = int(ori_h * ratio)
                new_w = test_size

            if new_h % 8 != 0:
                new_h = (int(new_h / 8)) * 8
            else:
                new_h = new_h
            if new_w % 8 != 0:
                new_w = (int(new_w / 8)) * 8
            else:
                new_w = new_w
            return new_h, new_w

        # Step 1: resize while keeping the h/w ratio. The largest side (i.e., height or width) is reduced to $size.
        #                                             The other is reduced accordingly
        test_size = self.size
        new_h, new_w = find_new_hw(image.shape[0], image.shape[1], test_size)

        image_crop = cv2.resize(image, dsize=(int(new_w), int(new_h)),
                                interpolation=cv2.INTER_LINEAR)

        # Step 2: Pad wtih 0 whatever needs to be padded to get a ($size, $size) image
        back_crop = np.zeros((test_size, test_size, 3))
        if self.padding:
            back_crop[:, :, 0] = self.padding[0]
            back_crop[:, :, 1] = self.padding[1]
            back_crop[:, :, 2] = self.padding[2]
        back_crop[:new_h, :new_w, :] = image_crop
        image = back_crop

        # Step 3: Do the same for the label (the padding is 255)
        if label is not None:
            s_mask = label
            new_h, new_w = find_new_hw(s_mask.shape[0], s_mask.shape[1], test_size)
            s_mask = cv2.resize(s_mask.astype(np.float32), dsize=(int(new_w), int(new_h)),
                                interpolation=cv2.INTER_NEAREST)
            back_crop_s_mask = np.ones((test_size, test_size)) * 255
            back_crop_s_mask[:new_h, :new_w] = s_mask
            label = back_crop_s_mask

            return image, label
        else:
            return image #, new_h, new_w