from imgaug import augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
import numpy as np
import torch


class ImgAugTransform:
    def __init__(self):
        # sometimes = lambda aug: iaa.Sometimes(0.5, aug)

        self.aug_brightness = iaa.Add((-30, 15))
        self.aug_flipub = iaa.Flipud(1.0)
        self.aug_blur = iaa.GaussianBlur(sigma=(0.1, 1.0))
        self.aug_fliplr = iaa.Fliplr(1.0)
        self.aug_affline_rot = iaa.Affine(rotate=(-45, 45), mode="wrap")
        self.aug_affline_she = iaa.Affine(shear=(-16, 16), mode="wrap")

    def __call__(self, img, label, aug=0):

        if aug == 0:
            segmap = SegmentationMapsOnImage(label, shape=label.shape)
            aug_img, aug_map = self.aug_affline_rot(
                image=img, segmentation_maps=segmap)

            aug_img = aug_img.astype(np.float32)/255
            aug_map = aug_map.arr

            aug_imgTensor = torch.from_numpy(aug_img.transpose((2, 0, 1)))
            aug_mapTensor = torch.from_numpy(aug_map.transpose((2, 0, 1)))

            return aug_imgTensor, aug_mapTensor[0, :, :]

        elif aug == 1:
            aug_img = self.aug_brightness(image=img)

            aug_img = aug_img.astype(np.float32)/255
            aug_map = label

            aug_imgTensor = torch.from_numpy(aug_img.transpose((2, 0, 1)))
            aug_mapTensor = torch.from_numpy(aug_map.transpose((2, 0, 1)))

            return aug_imgTensor, aug_mapTensor[0, :, :]

        elif aug == 2:
            aug_img = self.aug_blur(image=img)
            aug_img = aug_img.astype(np.float32)/255
            aug_map = label

            aug_imgTensor = torch.from_numpy(aug_img.transpose((2, 0, 1)))
            aug_mapTensor = torch.from_numpy(aug_map.transpose((2, 0, 1)))

            return aug_imgTensor, aug_mapTensor[0, :, :]

        elif aug == 3:
            aug_img = img.astype(np.float32)/255
            aug_map = label

            aug_imgTensor = torch.from_numpy(aug_img.transpose((2, 0, 1)))
            aug_mapTensor = torch.from_numpy(aug_map.transpose((2, 0, 1)))

            return aug_imgTensor, aug_mapTensor[0, :, :]

        elif aug == 4:
            segmap = SegmentationMapsOnImage(label, shape=label.shape)
            aug_img, aug_map = self.aug_affline_she(
                image=img, segmentation_maps=segmap)
            aug_img = aug_img.astype(np.float32)/255
            aug_map = aug_map.arr

            aug_imgTensor = torch.from_numpy(aug_img.transpose((2, 0, 1)))
            aug_mapTensor = torch.from_numpy(aug_map.transpose((2, 0, 1)))

            return aug_imgTensor, aug_mapTensor[0, :, :]
