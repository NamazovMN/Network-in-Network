from data_preparator import DataPreparator
from model import NIN
from train import Trainer
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

pca = PCA(n_components=1, random_state=0, svd_solver='randomized')

data_reader = DataPreparator("cifar_10", 128)
# train_loader, validation_loader, test_loader, trainset, testset, validationset = data_reader.prepare_data()
trainset, testset = data_reader.load_datasets()
print(np.array(trainset[12]))
# image = testset[12][0].numpy()
image = image.transpose([1,2,0])
# image_single = np.reshape(image, (1,3*32*32))

# pca.fit(image_single)

# np.vstack()
# pca.fit(image)
# print(image_single.shape)
def global_contrast_normalization(image_dataset, lmbd, epsilon, s):
    image = []
    for each in image_dataset:
        img_temp = each[0].numpy()
        image.append(img_temp.transpose([1,2,0]))
    image_contrast = []
    for each in image:
        image_mean = np.mean(each)
        image_data = each-image_mean

        contrast = np.sqrt(lmbd + np.mean(image_data**2))
        image_data = s*image_data/max(contrast, epsilon)
        image_contrast.append(image_data)
    image_contrast = np.array(image_contrast)
    return image_contrast
def ZCA_whitening(image):
    # image = []
    # for each in image_dataset:
    #     img_temp = each[0].numpy()
    #     image.append(img_temp.transpose([1,2,0]))
    # image = np.array(image)
    print(image.shape)
    # image = np.array(image)
    image = image.reshape(image.shape[0],image.shape[1]*image.shape[2]*image.shape[3])
    print(image.shape)
    image_norm = image/255
    image_norm = image_norm-image_norm.mean(axis = 0)
    image_cov = np.cov(image_norm, rowvar = False)
    print(image_cov.shape)
    U,S,V = np.linalg.svd(image_cov)
    epsilon = 0.1
    image_ZCA = U.dot(np.diag(1.0/np.sqrt(S+epsilon))).dot(U.T).dot(image_norm.T).T
    image_ZCA_rescaled = (image_ZCA-image_ZCA.min())/(image_ZCA.max()-image_ZCA.min())
    image_ZCA_rescaled = image_ZCA_rescaled.reshape(image.shape[0], 32,32,3)
    return image_ZCA_rescaled

image_ZCA = global_contrast_normalization(trainset,1,0.1,1)
image_ZCA = ZCA_whitening(image_ZCA)

for each in range(len(trainset)):
    trainset[each][0] = image_ZCA[each]
print(image_ZCA[12].shape)
figure, axes = plt.subplots(1,2)
axes[0].imshow(image)
axes[1].imshow(image_ZCA[12,:])
plt.show()