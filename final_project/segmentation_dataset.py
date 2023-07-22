"""Data utility functions."""
import os
import numpy as np
import torch
import torch.utils.data as data
from torchvision import transforms
from PIL import Image

import _pickle as pickle

SEG_LABELS_LIST = [
    {"id": 	0	, "name": "viod",       "rgb_values": [0, 0, 0]},
    {"id": 	1	, "name": "bed",        "rgb_values": [1, 0, 0]},
    {"id": 	2	, "name": "windowpane",    "rgb_values": [2, 0, 0]},
    {"id": 	3	, "name": "cabinet",      "rgb_values": [3, 0, 0]},
    {"id": 	4	, "name": "person",      "rgb_values": [4, 0, 0]},
    {"id": 	5	, "name": "door",       "rgb_values": [5, 0, 0]},
    {"id": 	6	, "name": "table",       "rgb_values": [6, 0, 0]},
    {"id": 	7	, "name": "curtain",      "rgb_values": [7, 0, 0]},
    {"id": 	8	, "name": "chair",       "rgb_values": [8, 0, 0]},
    {"id": 	9	, "name": "car",        "rgb_values": [9, 0, 0]},
    {"id": 	10	, "name": "painting",    "rgb_values": [10, 0, 0]},
    {"id": 	11	, "name": "sofa",      "rgb_values": [11, 0, 0]},
    {"id": 	12	, "name": "shelf",     "rgb_values": [12, 0, 0]},
    {"id": 	13	, "name": "mirror",     "rgb_values": [13, 0, 0]},
    {"id": 	14	, "name": "armchair",    "rgb_values": [14, 0, 0]},
    {"id": 	15	, "name": "seat",      "rgb_values": [15, 0, 0]},
    {"id": 	16	, "name": "fence",       "rgb_values": [	16	,   0,    0]},
    {"id": 	17	, "name": "desk	",       "rgb_values": [	17	,   0,    0]},
    {"id": 	18	, "name": "wardrobe	",       "rgb_values": [	18	,   0,    0]},
    {"id": 	19	, "name": "lamp	",       "rgb_values": [	19	,   0,    0]},
    {"id": 	20	, "name": "bathtub	",       "rgb_values": [	20	,   0,    0]},
    {"id": 	21	, "name": "railing	",       "rgb_values": [	21	,   0,    0]},
    {"id": 	22	, "name": "cushion	",       "rgb_values": [	22	,   0,    0]},
    {"id": 	23	, "name": "box	",       "rgb_values": [	23	,   0,    0]},
    {"id": 	24	, "name": "column	",       "rgb_values": [	24	,   0,    0]},
    {"id": 	25	, "name": "signboard	",       "rgb_values": [	25	,   0,    0]},
    {"id": 	26	, "name": "chest of drawers	",       "rgb_values": [	26	,   0,    0]},
    {"id": 	27	, "name": "counter	",       "rgb_values": [	27	,   0,    0]},
    {"id": 	28	, "name": "sink	",       "rgb_values": [	28	,   0,    0]},
    {"id": 	29	, "name": "fireplace	",       "rgb_values": [	29	,   0,    0]},
    {"id": 	30	, "name": "refrigerator	",       "rgb_values": [	30	,   0,    0]},
    {"id": 	31	, "name": "stairs	",       "rgb_values": [	31	,   0,    0]},
    {"id": 	32	, "name": "case	",       "rgb_values": [	32	,   0,    0]},
    {"id": 	33	, "name": "pool table	",       "rgb_values": [	33	,   0,    0]},
    {"id": 	34	, "name": "pillow	",       "rgb_values": [	34	,   0,    0]},
    {"id": 	35	, "name": "screen door	",       "rgb_values": [	35	,   0,    0]},
    {"id": 	36	, "name": "bookcase	",       "rgb_values": [	36	,   0,    0]},
    {"id": 	37	, "name": "coffee table	",       "rgb_values": [	37	,   0,    0]},
    {"id": 	38	, "name": "toilet	",       "rgb_values": [	38	,   0,    0]},
    {"id": 	39	, "name": "flower	",       "rgb_values": [	39	,   0,    0]},
    {"id": 	40	, "name": "book	",       "rgb_values": [	40	,   0,    0]},
    {"id": 	41	, "name": "bench	",       "rgb_values": [	41	,   0,    0]},
    {"id": 	42	, "name": "countertop	",       "rgb_values": [	42	,   0,    0]},
    {"id": 	43	, "name": "stove	",       "rgb_values": [	43	,   0,    0]},
    {"id": 	44	, "name": "palm	",       "rgb_values": [	44	,   0,    0]},
    {"id": 	45	, "name": "kitchen island	",       "rgb_values": [	45	,   0,    0]},
    {"id": 	46	, "name": "computer	",       "rgb_values": [	46	,   0,    0]},
    {"id": 	47	, "name": "swivel chair	",       "rgb_values": [	47	,   0,    0]},
    {"id": 	48	, "name": "boat	",       "rgb_values": [	48	,   0,    0]},
    {"id": 	49	, "name": "arcade machine	",       "rgb_values": [	49	,   0,    0]},
    {"id": 	50	, "name": "bus	",       "rgb_values": [	50	,   0,    0]},
    {"id": 	51	, "name": "towel	",       "rgb_values": [	51	,   0,    0]},
    {"id": 	52	, "name": "light	",       "rgb_values": [	52	,   0,    0]},
    {"id": 	53	, "name": "truck	",       "rgb_values": [	53	,   0,    0]},
    {"id": 	54	, "name": "chandelier	",       "rgb_values": [	54	,   0,    0]},
    {"id": 	55	, "name": "awning	",       "rgb_values": [	55	,   0,    0]},
    {"id": 	56	, "name": "streetlight	",       "rgb_values": [	56	,   0,    0]},
    {"id": 	57	, "name": "booth	",       "rgb_values": [	57	,   0,    0]},
    {"id": 	58	, "name": "television receiver	",       "rgb_values": [	58	,   0,    0]},
    {"id": 	59	, "name": "airplane	",       "rgb_values": [	59	,   0,    0]},
    {"id": 	60	, "name": "apparel	",       "rgb_values": [	60	,   0,    0]},
    {"id": 	61	, "name": "pole	",       "rgb_values": [	61	,   0,    0]},
    {"id": 	62	, "name": "bannister	",       "rgb_values": [	62	,   0,    0]},
    {"id": 	63	, "name": "ottoman	",       "rgb_values": [	63	,   0,    0]},
    {"id": 	64	, "name": "bottle	",       "rgb_values": [	64	,   0,    0]},
    {"id": 	65	, "name": "van	",       "rgb_values": [	65	,   0,    0]},
    {"id": 	66	, "name": "ship	",       "rgb_values": [	66	,   0,    0]},
    {"id": 	67	, "name": "fountain	",       "rgb_values": [	67	,   0,    0]},
    {"id": 	68	, "name": "washer	",       "rgb_values": [	68	,   0,    0]},
    {"id": 	69	, "name": "plaything	",       "rgb_values": [	69	,   0,    0]},
    {"id": 	70	, "name": "stool	",       "rgb_values": [	70	,   0,    0]},
    {"id": 	71	, "name": "barrel	",       "rgb_values": [	71	,   0,    0]},
    {"id": 	72	, "name": "basket	",       "rgb_values": [	72	,   0,    0]},
    {"id": 	73	, "name": "bag	",       "rgb_values": [	73	,   0,    0]},
    {"id": 	74	, "name": "minibike	",       "rgb_values": [	74	,   0,    0]},
    {"id": 	75	, "name": "oven	",       "rgb_values": [	75	,   0,    0]},
    {"id": 	76	, "name": "ball	",       "rgb_values": [	76	,   0,    0]},
    {"id": 	77	, "name": "food	",       "rgb_values": [	77	,   0,    0]},
    {"id": 	78	, "name": "step	",       "rgb_values": [	78	,   0,    0]},
    {"id": 	79	, "name": "trade name	",       "rgb_values": [	79	,   0,    0]},
    {"id": 	80	, "name": "microwave	",       "rgb_values": [	80	,   0,    0]},
    {"id": 	81	, "name": "pot	",       "rgb_values": [	81	,   0,    0]},
    {"id": 	82	, "name": "animal	",       "rgb_values": [	82	,   0,    0]},
    {"id": 	83	, "name": "bicycle	",       "rgb_values": [	83	,   0,    0]},
    {"id": 	84	, "name": "dishwasher	",       "rgb_values": [	84	,   0,    0]},
    {"id": 	85	, "name": "screen	",       "rgb_values": [	85	,   0,    0]},
    {"id": 	86	, "name": "sculpture	",       "rgb_values": [	86	,   0,    0]},
    {"id": 	87	, "name": "hood	",       "rgb_values": [	87	,   0,    0]},
    {"id": 	88	, "name": "sconce	",       "rgb_values": [	88	,   0,    0]},
    {"id": 	89	, "name": "vase	",       "rgb_values": [	89	,   0,    0]},
    {"id": 	90	, "name": "traffic light	",       "rgb_values": [	90	,   0,    0]},
    {"id": 	91	, "name": "tray	",       "rgb_values": [	91	,   0,    0]},
    {"id": 	92	, "name": "ashcan	",       "rgb_values": [	92	,   0,    0]},
    {"id": 	93	, "name": "fan	",       "rgb_values": [	93	,   0,    0]},
    {"id": 	94	, "name": "plate	",       "rgb_values": [	94	,   0,    0]},
    {"id": 	95	, "name": "monitor	",       "rgb_values": [	95	,   0,    0]},
    {"id": 	96	, "name": "bulletin board	",       "rgb_values": [	96	,   0,    0]},
    {"id": 	97	, "name": "radiator	",       "rgb_values": [	97	,   0,    0]},
    {"id": 	98	, "name": "glass	",       "rgb_values": [	98	,   0,    0]},
    {"id": 	99	, "name": "clock	",       "rgb_values": [	99	,   0,    0]},
    {"id": 	100 , "name": "flag	",       "rgb_values": [	100	,   0,    0]}]


def label_img_to_rgb(label_img):
    label_img = np.squeeze(label_img)
    labels = np.unique(label_img)
    label_infos = [l for l in SEG_LABELS_LIST if l['id'] in labels]

    label_img_rgb = np.array([label_img, label_img, label_img]).transpose(1,2,0)
    for l in label_infos:
        mask = label_img == l['id']
        label_img_rgb[mask] = l['rgb_values']

    return label_img_rgb.astype(np.uint8)


class SegmentationData(data.Dataset):

    def __init__(self, image_paths_file, folder, val_type):
        self.root_dir_name = os.path.dirname(image_paths_file)

        with open(image_paths_file) as f:
            self.image_names = f.read().splitlines()
        
        self.folder = folder
        self.val_type = val_type

    def __getitem__(self, key):
        if isinstance(key, slice):
            # get the start, stop, and step from the slice
            return [self[ii] for ii in range(*key.indices(len(self)))]
        elif isinstance(key, int):
            # handle negative indices
            if key < 0:
                key += len(self)
            if key < 0 or key >= len(self):
                raise IndexError("The index (%d) is out of range." % key)
            # get the data from direct index
            return self.get_item_from_index(key)
        else:
            raise TypeError("Invalid argument type.")

    def __len__(self):
        return len(self.image_names)

    def get_item_from_index(self, index):
        to_tensor = transforms.ToTensor()
        img_name = self.image_names[index]

        img = Image.open(os.path.join(self.root_dir_name, 'images', self.folder, img_name + '.jpg')).convert('RGB')

        width, height = img.size

        center_crop = transforms.CenterCrop(240)
        img = center_crop(img)
        img = to_tensor(img)

        target = Image.open(os.path.join(self.root_dir_name, 'annotations_instance', self.folder, img_name + self.val_type))
        target = center_crop(target)
        target = np.array(target, dtype=np.int64)
        target_labels = target[..., 0]

        for label in SEG_LABELS_LIST:
            mask = np.all(target[..., 0] == label['rgb_values'][0], axis=(0, 1))
            target_labels[mask] = label['id']

        target_labels = torch.from_numpy(target_labels.copy())

        return img, target_labels, width, height, img_name
