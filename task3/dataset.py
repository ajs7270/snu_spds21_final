import os

import numpy as np
import torch
import torch.nn as nn

import cv2
import natsort

from skimage.color import rgb2gray, rgba2rgb
import imageio

# Data Loader
class CustomDataset(torch.utils.data.Dataset):
  def __init__(self, data_dir, transform=None, is_train=True):#fdir, pdir, sdir, transform=None):
    self.is_train = is_train
    self.aug_dir = '../Data/images/'

    self.fist_dir = os.path.join(data_dir,'rock/')
    self.palm_dir = os.path.join(data_dir,'paper/')
    self.swing_dir = os.path.join(data_dir,'scissors/')

    self.transform = transform

    lst_fist = os.listdir(self.fist_dir)
    lst_palm = os.listdir(self.palm_dir)
    lst_swing = os.listdir(self.swing_dir)
    lst_aug = os.listdir(self.aug_dir)

    lst_fist = [f for f in lst_fist if f.endswith((".jpg",'.png'))]
    lst_palm = [f for f in lst_palm if f.endswith((".jpg",'.png'))]
    lst_swing = [f for f in lst_swing if f.endswith((".jpg",'png'))]
    lst_aug = [self.aug_dir + f for f in lst_aug if f.endswith((".jpg", 'png'))]

    self.lst_dir = [self.fist_dir] * len(lst_fist) + [self.palm_dir] * len(lst_palm) + [self.swing_dir] * len(lst_swing)
    self.lst_prs = natsort.natsorted(lst_fist) + natsort.natsorted(lst_palm) + natsort.natsorted(lst_swing)
    if self.is_train:
      self.lst_aug = self.load_augmentation(lst_aug)
    else:
      self.lst_aug = range(1)
  def __len__(self):
    return len(self.lst_prs)

  def __getitem__(self, index): 
    self.img_dir = self.lst_dir[index]
    self.img_name = self.lst_prs[index]

    return [self.img_dir, self.img_name] 

  def load_augmentation(self,data):
    result = []
    for path in data:
      prs_img = imageio.imread(os.path.join(path))

      resized_img = cv2.resize(prs_img,(89,100))
      result.append(resized_img)
    return result

  def custom_collate_fn(self, data):
    aug_datas = [{'input':[] , 'label':[], 'filename':[]} for i in range(len(self.lst_aug))]

    for sample in data:
      try :
        prs_img = imageio.imread(os.path.join(sample[0] + sample[1]))
      except :
        continue
      resized_img = cv2.resize(prs_img,(89,100))
      for aug_idx, background_img in enumerate(self.lst_aug):
        gray_img = None
        if self.is_train:
          # --② 마스크 생성, 합성할 이미지 전체 영역을 255로 셋팅
          mask = np.full_like(resized_img, 255)

          # --③ 합성 대상 좌표 계산(img2의 중앙)
          height, width, _ = background_img.shape
          center = (width // 2, height // 2)

          # --④ seamlessClone 으로 합성
          aug_img = cv2.seamlessClone(resized_img, background_img, mask, center, cv2.NORMAL_CLONE)
          gray_img = rgb2gray(aug_img)
        else :
          gray_img = rgb2gray(resized_img)

        if gray_img.ndim == 2:
          gray_img = gray_img[:, :, np.newaxis]

        aug_datas[aug_idx]['input'].append(gray_img.reshape(89, 100, 1))
        dir_split = sample[0].split('/')
        if dir_split[-2] in ('r','rock'):
          aug_datas[aug_idx]['label'].append(np.array(1))
        elif dir_split[-2] in ('p','paper'):
          aug_datas[aug_idx]['label'].append(np.array(0))
        elif dir_split[-2] in ('s','scissors'):
          aug_datas[aug_idx]['label'].append(np.array(2))
        aug_datas[aug_idx]['filename'].append(sample[1])

    result = []
    for aug_data in aug_datas:
      if self.transform:
        result.append(self.transform(aug_data))

    if self.is_train == False:
      result = result[0]

    return result


class ToTensor(object):
  def __call__(self, data):
    filename, label, input = data['filename'], data['label'], data['input']

    input_tensor = torch.empty(len(input),89,100)
    label_tensor = torch.empty(len(input))
    for i in range(len(input)):
      input[i] = input[i].transpose((2, 0, 1)).astype(np.float32)
      input_tensor[i] = torch.from_numpy(input[i])
      label_tensor[i] = torch.from_numpy(label[i])
    input_tensor = torch.unsqueeze(input_tensor, 1)

    data = {'filename': filename, 'label': label_tensor.long(), 'input': input_tensor}

    return data

