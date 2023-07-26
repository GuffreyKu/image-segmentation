import csv
import sys
import random
import cv2
from gaskLibs.utils.segaug import ImgAugTransform
from torch.utils.data.dataset import Dataset

sys.path.insert(0, '..')


class ImageDataset(Dataset):
    def __init__(self, csv_path, img_size, is_aug):
        super().__init__()
        self.img_path = []
        self.labels = []
        self.transform = ImgAugTransform()
        self.isaug = is_aug
        self.img_w = img_size[0]
        self.img_h = img_size[1]

        with open(csv_path, newline='') as csvfile:
            rows = csv.reader(csvfile)
            header = next(rows)
            for row in rows:
                self.img_path.append(row[0])
                self.labels.append(row[1])

    def __getitem__(self, index):
        # print(self.img_path[index])
        # print(self.labels[index])

        img = cv2.imread(self.img_path[index])
        img = cv2.resize(img, (self.img_w, self.img_h))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img_map = cv2.imread(self.labels[index])
        img_map = cv2.resize(img_map, (self.img_w, self.img_h))
        # pixel label to class label
        img_map = img_map / 255
        img_map = img_map.astype(int)

        if self.isaug:
            j = random.randint(0, 4)
            img_tensor, map_tensor = self.transform(img, img_map, j)

        else:
            img_tensor, map_tensor = self.transform(img, img_map, 3)

        return img_tensor, map_tensor

    def __len__(self):
        return len(self.labels)
