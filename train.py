import torch.nn as nn
import torch
from tqdm import tqdm


class Trainer(object):
    def __init__(self, optimizer, loss_function, train_dataset, test_dataset, validation_dataset):
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.train_loader = train_dataset
        self.test_loader = test_dataset
        self.validation_loader = validation_dataset

    def train_phase(self, epochs, model):
        train_accuracy_list = []
        train_loss_list = []
        test_accuracy_list = []
        test_loss_list = []
        train_loss = 0
        train_accuracy = 0
        for epoch in range(epochs):
            model.train()
            epoch_loss = 0
            total_train = 0
            correct_train = 0
            for i, (train, labels) in enumerate(tqdm(self.train_loader,0)):
                train = train.cuda()
                labels = labels.cuda()
                # print(train.shape)
                self.optimizer.zero_grad()
                outputs = model(train)
                # outputs = model(train.permute(0,3,1,2))

                sample_loss = self.loss_function(outputs, labels)
                sample_loss.backward()

                self.optimizer.step()
                epoch_loss += sample_loss.tolist()


                #train accuracy

                _, prediction = torch.max(outputs.data, 1)
                total_train += labels.nelement()
                correct_train += prediction.eq(labels.data).sum().item()
            
            train_loss_epoch = epoch_loss/len(self.train_loader)

            train_accuracy_epoch = correct_train/total_train

            test_loss_epoch = self.evaluate(model)

            test_accuracy_epoch = self.compute_accuracy(model)                

            
            train_loss += train_loss_epoch
            
            train_accuracy += train_accuracy_epoch
            
            train_accuracy_list.append(train_accuracy_epoch)
            
            train_loss_list.append(train_loss_epoch)
            
            test_accuracy_list.append(test_accuracy_epoch)
            
            test_loss_list.append(test_loss_epoch)
            
            
            
            
            print('Epoch: {}  Train Loss: {:0.4f}  Train Accuracy: {:0.4f}  Validation Loss: {:0.4f} Validation Accuracy: {:0.4f}'.format(epoch+1, train_loss_epoch, train_accuracy_epoch, test_loss_epoch, test_accuracy_epoch ))
            self.adjust_learning_rate(epoch)
        train_loss_avg = train_loss/epochs
        train_accuracy_avg = train_accuracy/epochs
        return train_loss_avg, train_accuracy_avg, train_loss_list, train_accuracy_list, test_loss_list, test_accuracy_list

    def evaluate(self, model, validation = True):
        if validation:
            dataset = self.validation_loader
        else:
            dataset = self.test_loader
        losses = 0
        model.eval()
        with torch.no_grad():
            for i,(input_data, label_data) in enumerate(dataset,0):
                input_data = input_data.cuda()
                label_data = label_data.cuda()
                output_data = model(input_data)
                
                # output_data = model(input_data.permute(0,3,1,2))
                loss = self.loss_function(output_data, label_data)
                losses += loss.tolist()

        return losses/len(dataset)

    def compute_accuracy(self, model, validation = True):
        
        if validation:
            dataset = self.validation_loader
        else:
            dataset = self.test_loader

        with torch.no_grad():
            # Calculate Accuracy
            correct = 0
            total = 0
            # Iterate through test dataset
            for i, (test,labels) in enumerate(dataset, 0):
                test = test.cuda() 
                labels = labels.cuda()

                # Forward propagation
                outputs = model(test)

                predicted = torch.max(outputs.data, 1)[1]

                # Total number of labels
                total += len(labels)

                correct += (predicted == labels).sum()

            accuracy = correct / float(total)
            return accuracy
    
    def adjust_learning_rate(self, epoch):
        if epoch!= 0 and epoch%40==0:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] =0.1*param_group['lr'] 
            print("Learning rate is 10 percent of the previous learning rate")
