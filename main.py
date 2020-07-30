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
dataset_name = "svhn"

print("Chosen dataset is {}".format(dataset_name))
data_reader = DataPreparator(dataset_name, 128)
train_loader, validation_loader, test_loader= data_reader.prepare_dataloaders()



model = NIN(10).cuda()
summary(model, (3,32,32))

optimizer = optim.Adam(model.parameters(), lr = 0.0001)
loss_function = nn.CrossEntropyLoss()

trainer = Trainer(optimizer, loss_function, train_loader, test_loader, validation_loader)
train_loss, train_accuracy, train_loss_list, train_accuracy_list, test_loss_list, test_accuracy_list = trainer.train_phase(100, model)

torch.save(model, "NIN_SVHN_w_dropout")

model_load = torch.load("NIN_SVHN_w_dropout")
model_load.eval()
test_accuracy = trainer.compute_accuracy(model_load, validation = False)
print("Test accuracy is {}".format(test_accuracy))

plt.plot(list(range(1,101)), train_accuracy_list, list(range(1,101)), test_accuracy_list)
plt.title("Accuracy results SVHN")
plt.xlabel("Epochs")
plt.ylabel("Accuracy rate")
plt.legend(['train', 'validation'], loc = 'upper left')
plt.savefig("Accuracies_SVHN_0.7.png")
plt.show()


plt.plot(list(range(1,101)), train_loss_list, list(range(1,101)), test_loss_list)
plt.title("Error results SVHN")
plt.xlabel("Epochs")
plt.ylabel("Error Rate")
plt.legend(['train', 'validation'], loc = 'upper left')
plt.savefig("Losses_SVHN_0.7.png")
plt.show()
