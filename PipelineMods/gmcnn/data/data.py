import torch
import numpy as np
import cv2
import os
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import random

class ToTensor(object):
    def __call__(self, sample):
        entry = {}
        for k in sample:
            if k == 'rect':
                entry[k] = torch.IntTensor(sample[k])
            else:
                entry[k] = torch.FloatTensor(sample[k])
        return entry


class InpaintingDataset(Dataset):
    def __init__(self, info_list, root_dir='', im_size=(256, 256), transform=None):
        self.filenames = open(info_list, 'rt').read().splitlines()
        self.root_dir = root_dir
        self.transform = transform
        self.im_size = im_size
        np.random.seed(2018)

    def __len__(self):
        return len(self.filenames)

    def read_image(self, filepath):
        image = cv2.imread(filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, c = image.shape
        if h != self.im_size[0] or w != self.im_size[1]:
            ratio = max(1.0*self.im_size[0]/h, 1.0*self.im_size[1]/w)
            im_scaled = cv2.resize(image, None, fx=ratio, fy=ratio)
            h, w, _ = im_scaled.shape
            h_idx = (h-self.im_size[0]) // 2
            w_idx = (w-self.im_size[1]) // 2
            im_scaled = im_scaled[h_idx:h_idx+self.im_size[0], w_idx:w_idx+self.im_size[1],:]
            im_scaled = np.transpose(im_scaled, [2, 0, 1])
        else:
            im_scaled = np.transpose(image, [2, 0, 1])
        return im_scaled

    def __getitem__(self, idx):
        image = self.read_image(os.path.join(self.root_dir, self.filenames[idx]))
        #print("image path:", os.path.join(self.root_dir, self.filenames[idx]))
        sample = {'gt': image}
        if self.transform:
            sample = self.transform(sample)
        return sample

#============================
# Modified versiion by Vajira
# To load image and mask from segmented images of Hyper-kvasir
#============================
class InpaintingDataset_WithMask(Dataset):
    def __init__(self, info_list, root_dir='', im_size=(256, 256), transform=None):
        self.filenames= open(info_list, 'rt').read().splitlines()
       
        self.root_dir = root_dir
        
        self.root_dir_img = os.path.join(self.root_dir, "images")
        self.root_dir_mask = os.path.join(self.root_dir, "masks")

        self.transform = transform
        self.im_size = im_size
        np.random.seed(2018)

    def __len__(self):
        return len(self.filenames)

    def read_image(self, filepath):
        #print(filepath)
        image = cv2.imread(filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #print(image.shape)
        h, w, c = image.shape
        if h != self.im_size[0] or w != self.im_size[1]:
            ratio = max(1.0*self.im_size[0]/h, 1.0*self.im_size[1]/w)
            im_scaled = cv2.resize(image, None, fx=ratio, fy=ratio)
            #print(im_scaled.shape)
            h, w, _ = im_scaled.shape
            h_idx = (h-self.im_size[0]) // 2
            w_idx = (w-self.im_size[1]) // 2
            im_scaled = im_scaled[h_idx:h_idx+self.im_size[0], w_idx:w_idx+self.im_size[1],:]
            plt.imsave("test_img.jpeg",im_scaled)
            im_scaled = np.transpose(im_scaled, [2, 0, 1])
        else:
            im_scaled = np.transpose(image, [2, 0, 1])
            #print("This is running")
        return im_scaled

    # added by vajira
    # To read mask
    def read_mask(self, filepath):
        #print(filepath)
        image = cv2.imread(filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #image = np.expand_dims(image, axis=2)
        #print(image.shape)
        #image = np.where(image > 0, 1, 0)
       # print(image.shape)
        h, w = image.shape
        if h != self.im_size[0] or w != self.im_size[1]:
            ratio = max(1.0*self.im_size[0]/h, 1.0*self.im_size[1]/w)
            im_scaled = cv2.resize(image, None, fx=ratio, fy=ratio)
            #print(im_scaled.shape)
            h, w = im_scaled.shape
            h_idx = (h-self.im_size[0]) // 2
            w_idx = (w-self.im_size[1]) // 2
            im_scaled = im_scaled[h_idx:h_idx+self.im_size[0], w_idx:w_idx+self.im_size[1]]

            im_scaled = np.expand_dims(im_scaled, axis=2)

            #plt.imsave("test_mask.jpeg",im_scaled, cmap="gray")
            im_scaled = np.transpose(im_scaled, [2, 0, 1])
            im_scaled = np.where(im_scaled > 0, 1, 0) # convert into 0 and 1
        else:
            im_scaled = np.expand_dims(im_scaled, axis=2)
            im_scaled = np.transpose(image, [2, 0, 1])
            im_scaled = np.where(im_scaled > 0, 1, 0) # convert into 0 and 1
            #print("This is running")
        return im_scaled

    def __getitem__(self, idx):
        image = self.read_image(os.path.join(self.root_dir_img, self.filenames[idx]))
        mask = self.read_mask(os.path.join(self.root_dir_mask, self.filenames[idx]))
        # print(mask.shape)
        #print("image path:", os.path.join(self.root_dir_img, self.filenames[idx]))
        #print(self.filenames[idx])
        sample = {'gt': image, 'mask' : mask}
        if self.transform:
            sample = self.transform(sample)
        return sample

if __name__ == "__main__":
    file_list = "/work/vajira/DATA/hyper_kvasir/data_new/segmented/data/segmented-images/file_list.flist"
    data_root = "/work/vajira/DATA/hyper_kvasir/data_new/segmented/data/segmented-images" 
    ds = InpaintingDataset_WithMask(file_list, data_root)

    data_point = ds[500]
    print(data_point["gt"].shape)
    #print(data_point["mask"].tolist())

    #with open("test.txt", "w") as f:
    #    f.write(str(data_point["mask"]))
  #  plt.imshow(ds[0]["gt"].
    print(len(ds))


#============================
# Modified versiion by Vajira
# To load image and mask from segmented images of Hyper-kvasir
# Version 2: this has been changed to remove noise around the contour
#============================
class InpaintingDataset_WithMask_v2(Dataset):
    def __init__(self, info_list, root_dir='', im_size=(256, 256), transform=None):
        self.filenames= open(info_list, 'rt').read().splitlines()
       
        self.root_dir = root_dir
        
        self.root_dir_img = os.path.join(self.root_dir, "images")
        self.root_dir_mask = os.path.join(self.root_dir, "masks")

        self.transform = transform
        self.im_size = im_size
        np.random.seed(2018)

    def __len__(self):
        return len(self.filenames)

    def read_image(self, filepath):
        #print(filepath)
        image = cv2.imread(filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #print(image.shape)
        h, w, c = image.shape
        if h != self.im_size[0] or w != self.im_size[1]:
            ratio = max(1.0*self.im_size[0]/h, 1.0*self.im_size[1]/w)
            im_scaled = cv2.resize(image, None, fx=ratio, fy=ratio)
            #print(im_scaled.shape)
            h, w, _ = im_scaled.shape
            h_idx = (h-self.im_size[0]) // 2
            w_idx = (w-self.im_size[1]) // 2
            im_scaled = im_scaled[h_idx:h_idx+self.im_size[0], w_idx:w_idx+self.im_size[1],:]
            plt.imsave("test_img.jpeg",im_scaled)
            im_scaled = np.transpose(im_scaled, [2, 0, 1])
        else:
            im_scaled = np.transpose(image, [2, 0, 1])
            #print("This is running")
        return im_scaled

    # added by vajira
    # To read mask
    def read_mask(self, filepath):
        #print(filepath)
        image = cv2.imread(filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #image = np.expand_dims(image, axis=2)
        #print(image.shape)
        #image = np.where(image > 0, 1, 0)
       # print(image.shape)
        h, w = image.shape
        if h != self.im_size[0] or w != self.im_size[1]:
            ratio = max(1.0*self.im_size[0]/h, 1.0*self.im_size[1]/w)
            im_scaled = cv2.resize(image, None, fx=ratio, fy=ratio)
            #print(im_scaled.shape)
            h, w = im_scaled.shape
            h_idx = (h-self.im_size[0]) // 2
            w_idx = (w-self.im_size[1]) // 2
            im_scaled = im_scaled[h_idx:h_idx+self.im_size[0], w_idx:w_idx+self.im_size[1]]

            im_scaled = np.expand_dims(im_scaled, axis=2)

            #plt.imsave("test_mask.jpeg",im_scaled, cmap="gray")
            im_scaled = np.transpose(im_scaled, [2, 0, 1])
            im_scaled = np.where(im_scaled > 127, 1, 0) # convert into 0 and 1
        else:
            im_scaled = np.expand_dims(im_scaled, axis=2)
            im_scaled = np.transpose(image, [2, 0, 1])
            im_scaled = np.where(im_scaled > 127, 1, 0) # convert into 0 and 1
            #print("This is running")
        return im_scaled

    def __getitem__(self, idx):
        image = self.read_image(os.path.join(self.root_dir_img, self.filenames[idx]))
        mask = self.read_mask(os.path.join(self.root_dir_mask, self.filenames[idx]))
        # print(mask.shape)
        #print("image path:", os.path.join(self.root_dir_img, self.filenames[idx]))
        #print(self.filenames[idx])
        sample = {'gt': image, 'mask' : mask}
        if self.transform:
            sample = self.transform(sample)
        return sample


#============================
# Modified versiion by Vajira
# To load image and mask from segmented images of Hyper-kvasir
# Version 2: this has been changed to remove noise around the contour
#============================
class InpaintingDataset_WithMask_BB(Dataset):
    def __init__(self, info_list, root_dir='', im_size=(256, 256), transform=None):
        self.filenames= open(info_list, 'rt').read().splitlines()
       
        self.root_dir = root_dir
        
        self.root_dir_img = os.path.join(self.root_dir, "images")
        self.root_dir_mask = os.path.join(self.root_dir, "masks")

        self.transform = transform
        self.im_size = im_size
        np.random.seed(2018)

    def __len__(self):
        return len(self.filenames)

    def read_image(self, filepath):
        #print(filepath)
        image = cv2.imread(filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #print(image.shape)
        h, w, c = image.shape
        if h != self.im_size[0] or w != self.im_size[1]:
            ratio = max(1.0*self.im_size[0]/h, 1.0*self.im_size[1]/w)
            im_scaled = cv2.resize(image, None, fx=ratio, fy=ratio)
            #print(im_scaled.shape)
            h, w, _ = im_scaled.shape
            h_idx = (h-self.im_size[0]) // 2
            w_idx = (w-self.im_size[1]) // 2
            im_scaled = im_scaled[h_idx:h_idx+self.im_size[0], w_idx:w_idx+self.im_size[1],:]
            plt.imsave("test_img.jpeg",im_scaled)
            im_scaled = np.transpose(im_scaled, [2, 0, 1])
        else:
            im_scaled = np.transpose(image, [2, 0, 1])
            #print("This is running")
        return im_scaled

    # added by vajira

    # Conver mask image to BB
    def mask_to_bb(self, image): #image= gray scaled image
        ret, thresh = cv2.threshold(image, 127, 255, 0)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        contours_poly = [None]*len(contours)
        boundRect = [None]*len(contours)

        for i, c in enumerate(contours):
            contours_poly[i] = cv2.approxPolyDP(c, 3, True)
            boundRect[i] = cv2.boundingRect(contours_poly[i])
            #centers[i], radius[i] = cv.minEnclosingCircle(contours_poly[i])

        im_bb = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

        for i in range(len(contours)):
            
            cv2.rectangle(im_bb, (int(boundRect[i][0]), int(boundRect[i][1])), \
            (int(boundRect[i][0]+boundRect[i][2]), int(boundRect[i][1]+boundRect[i][3])), (255,255, 255), cv2.FILLED)

        return im_bb



    # To read mask
    def read_mask(self, filepath):
        #print(filepath)
        image = cv2.imread(filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        image = self.mask_to_bb(image) #convert mask into BB

        #image = np.expand_dims(image, axis=2)
        #print(image.shape)
        #image = np.where(image > 0, 1, 0)
       # print(image.shape)
        h, w = image.shape
        if h != self.im_size[0] or w != self.im_size[1]:
            ratio = max(1.0*self.im_size[0]/h, 1.0*self.im_size[1]/w)
            im_scaled = cv2.resize(image, None, fx=ratio, fy=ratio)
            #print(im_scaled.shape)
            h, w = im_scaled.shape
            h_idx = (h-self.im_size[0]) // 2
            w_idx = (w-self.im_size[1]) // 2
            im_scaled = im_scaled[h_idx:h_idx+self.im_size[0], w_idx:w_idx+self.im_size[1]]

            im_scaled = np.expand_dims(im_scaled, axis=2)

            #plt.imsave("test_mask.jpeg",im_scaled, cmap="gray")
            im_scaled = np.transpose(im_scaled, [2, 0, 1])
            im_scaled = np.where(im_scaled > 127, 1, 0) # convert into 0 and 1
        else:
            im_scaled = np.expand_dims(im_scaled, axis=2)
            im_scaled = np.transpose(image, [2, 0, 1])
            im_scaled = np.where(im_scaled > 127, 1, 0) # convert into 0 and 1
            #print("This is running")
        return im_scaled

    def __getitem__(self, idx):
        image = self.read_image(os.path.join(self.root_dir_img, self.filenames[idx]))
        mask = self.read_mask(os.path.join(self.root_dir_mask, self.filenames[idx]))
        # print(mask.shape)
        #print("image path:", os.path.join(self.root_dir_img, self.filenames[idx]))
        #print(self.filenames[idx])
        sample = {'gt': image, 'mask' : mask}
        if self.transform:
            sample = self.transform(sample)
        return sample



#============================
# Modified versiion by Vajira
# To load image from an image folder and random mask from given mask folders
# Version 2: this has been changed to remove noise around the contour
#============================
class InpaintingDataset_RandomPolypMask(Dataset):
    def __init__(self, info_list, mask_list,im_size=(256, 256), transform=None):
        self.filenames= open(info_list, 'rt').read().splitlines()
        self.masknames = open(mask_list, 'rt').read().splitlines()
       
        #self.root_dir = root_dir
        
        #self.root_dir_img = os.path.join(self.root_dir, "images")
        #self.root_dir_mask = os.path.join(self.root_dir, "masks")

        self.transform = transform
        self.im_size = im_size
        np.random.seed(2018)

    def __len__(self):
        return len(self.filenames)

    def read_image(self, filepath):
        #print(filepath)
        image = cv2.imread(filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #print(image.shape)
        h, w, c = image.shape
        if h != self.im_size[0] or w != self.im_size[1]:
            ratio = max(1.0*self.im_size[0]/h, 1.0*self.im_size[1]/w)
            im_scaled = cv2.resize(image, None, fx=ratio, fy=ratio)
            #print(im_scaled.shape)
            h, w, _ = im_scaled.shape
            h_idx = (h-self.im_size[0]) // 2
            w_idx = (w-self.im_size[1]) // 2
            im_scaled = im_scaled[h_idx:h_idx+self.im_size[0], w_idx:w_idx+self.im_size[1],:]
            plt.imsave("test_img.jpeg",im_scaled)
            im_scaled = np.transpose(im_scaled, [2, 0, 1])
        else:
            im_scaled = np.transpose(image, [2, 0, 1])
            #print("This is running")
        return im_scaled

    # added by vajira
    # To read mask
    def read_mask(self, filepath):
        #print(filepath)
        image = cv2.imread(filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #image = np.expand_dims(image, axis=2)
        #print(image.shape)
        #image = np.where(image > 0, 1, 0)
       # print(image.shape)
        h, w = image.shape
        if h != self.im_size[0] or w != self.im_size[1]:
            ratio = max(1.0*self.im_size[0]/h, 1.0*self.im_size[1]/w)
            im_scaled = cv2.resize(image, None, fx=ratio, fy=ratio)
            #print(im_scaled.shape)
            h, w = im_scaled.shape
            h_idx = (h-self.im_size[0]) // 2
            w_idx = (w-self.im_size[1]) // 2
            im_scaled = im_scaled[h_idx:h_idx+self.im_size[0], w_idx:w_idx+self.im_size[1]]

            im_scaled = np.expand_dims(im_scaled, axis=2)

            #plt.imsave("test_mask.jpeg",im_scaled, cmap="gray")
            im_scaled = np.transpose(im_scaled, [2, 0, 1])
            im_scaled = np.where(im_scaled > 127, 1, 0) # convert into 0 and 1
        else:
            im_scaled = np.expand_dims(im_scaled, axis=2)
            im_scaled = np.transpose(image, [2, 0, 1])
            im_scaled = np.where(im_scaled > 127, 1, 0) # convert into 0 and 1
            #print("This is running")
        return im_scaled

    def __getitem__(self, idx):
        image = self.read_image(self.filenames[idx])

        rand_mask_idx = random.randint(0, len(self.masknames) - 1)
        mask = self.read_mask(self.masknames[rand_mask_idx])
        # print(mask.shape)
        #print("image path:", os.path.join(self.root_dir_img, self.filenames[idx]))
        #print(self.filenames[idx])
        sample = {'gt': image, 'mask' : mask}
        if self.transform:
            sample = self.transform(sample)
        return sample





if __name__ == "__main__":
    file_list = "/work/vajira/DATA/hyper_kvasir/data_new/unlabelled/file_list.flist"
    mask_list = "/work/vajira/DATA/generated_polyp_masks/collected_mask_list_4k/mask_list.flist"
    data_root = "/work/vajira/DATA/hyper_kvasir/data_new/segmented/data/segmented-images" 
    ds = InpaintingDataset_RandomPolypMask(file_list, mask_list)#InpaintingDataset_WithMask(file_list, data_root)

    data_point = ds[500]
    print(data_point["gt"].shape)
    print(data_point["mask"].shape)
    #print(data_point["mask"].tolist())

    #with open("test.txt", "w") as f:
    #    f.write(str(data_point["mask"]))
  #  plt.imshow(ds[0]["gt"].
    print(len(ds))
