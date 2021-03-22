import os


gpu_specify = input ('Please choose a device, 0~3 for single GPU, 4 for all GPUs, none for CPU: \n')
if gpu_specify == '':
    gpu_sign = 0
elif gpu_specify == '4':
    gpu_sign = 1
elif gpu_specify == '0' or gpu_specify == '1' or gpu_specify == '2' or gpu_specify == '3':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_specify
    gpu_sign = 2
else:
    raise ValueError('incorrect GPU symbol')

import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
from torchsummary import summary
from torch.autograd import Variable
from torch.optim.sgd import SGD
import pickle
import alpha_optimizers
import os
import numpy as np
from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig
from DNN_models import CNN, Full_connect
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image

# Hyper Parameters
num_classes = 10
num_epochs = 10
batch_size = 32

#dataset_sign = int(input('Please input a dataset sign, 0 for mnist, 1 for cifar10, 2 for imagenet'))
model_sign = int(input('please input model sign: \n 0 for full_connect, 1 for CNN \nmodel_sign:'))
train_data = datasets.MNIST(root = 'data/mnist', train = True, transform = transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=0)
test_data = datasets.MNIST(root = 'data/mnist',train = False,transform = transforms.ToTensor())
test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False, num_workers=0)

#testing functon
def training(model_sign=0, optimizer_sign=0, learning_rate=0.01, momentum=0.9, beta=0.99, alpha=1):
    training_data = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    if model_sign == 0:
        net = Full_connect(num_classes)
    elif model_sign == 1:
        net = CNN(num_classes)
    else:
        raise ValueError('Not correct model sign')
    if gpu_sign == 1:
        net = torch.nn.DataParallel(net, device_ids=[0, 1, 2, 3])
    if gpu_sign != 0:
        net.cuda()
    net.train()
    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    print('optimizer_sign:' + str(optimizer_sign))
    if optimizer_sign == 0:
        optimizer = alpha_optimizers.Adamoptimizer(net.parameters(), lr=learning_rate, weight_decay=0.0001,
                                                   momentum=momentum, beta=beta)
    elif optimizer_sign == 1:
        optimizer = alpha_optimizers.SGD_momentumoptimizer(net.parameters(), lr=learning_rate, weight_decay=0.0001, momentum=momentum)
    elif optimizer_sign == 2:
        optimizer = alpha_optimizers.Adaboundoptimizer(net.parameters(), lr=learning_rate[0],sgd_lr=learning_rate[1], weight_decay=0.0001,
                                                                momentum=momentum, beta=beta, alpha = alpha)
    elif optimizer_sign == 3:
        optimizer = alpha_optimizers.lb_Adamoptimizer(net.parameters(), lr=learning_rate, weight_decay=0.0001,
                                                   momentum=momentum, beta=beta)
    elif optimizer_sign == 4:
        optimizer = alpha_optimizers.lb_SGDoptimizer(net.parameters(), lr=learning_rate, weight_decay=0.0001, momentum=momentum)
    elif optimizer_sign == 5:
        optimizer = alpha_optimizers.direction_Adamoptimizer(net.parameters(), lr=learning_rate, weight_decay=0.0001,
                                                               momentum=momentum, beta=beta)
    elif optimizer_sign == 6:
        optimizer = alpha_optimizers.direction_lbAdamoptimizer(net.parameters(), lr=learning_rate, weight_decay=0.0001,
                                                   momentum=momentum, beta=beta)
    elif optimizer_sign == 7:
        optimizer = alpha_optimizers.direction_Adamtosgdoptimizer(net.parameters(), lr=learning_rate, weight_decay=0.0001,
                                                               momentum=momentum, beta=beta)
    else:
        raise ValueError('Not correct algorithm symbol')

    # Train the Model
    for epoch in range(num_epochs):

        train_loss_log = AverageMeter()
        train_acc_log = AverageMeter()
        val_loss_log = AverageMeter()
        val_acc_log = AverageMeter()
        for i, (images, labels) in enumerate(train_loader):
            # Convert torch tensor to Variable
            if gpu_sign != 0:
                images = images.cuda()
                labels = labels.cuda()
            images = images.float()
            labels = Variable(labels).long()
            if model_sign == 0:
                images = images.view(-1, 784)
            # Forward + Backward + Optimize
            optimizer.zero_grad()  # zero the gradient buffer
            outputs = net(images)
            train_loss = criterion(outputs, labels)
            train_loss.backward()
            optimizer.step()
            prec1, prec5 = accuracy(outputs.data, labels.data, topk=(1, 5))
            train_loss_log.update(train_loss.data, images.size(0))
            train_acc_log.update(prec1, images.size(0))

            if (i + 1) % 20 == 0:
                print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f, Acc: %.8f'
                      % (epoch + 1, num_epochs, i + 1, 60000 / batch_size, train_loss_log.avg,
                         train_acc_log.avg))
                training_data['train_loss'].append(train_loss_log.avg.detach().cpu().numpy())
                training_data['train_acc'].append(train_acc_log.avg.detach().cpu().numpy())
        # Test the Model
        for images, labels in test_loader:
            if gpu_sign != 0:
                images = images.cuda()
                labels = labels.cuda()
            images = images.float()
            labels = Variable(labels).long()
            if model_sign == 0:
                images = images.view(-1, 784)
            outputs = net(images)
            test_loss = criterion(outputs, labels)
            val_loss_log.update(test_loss.data, images.size(0))
            prec1, prec5 = accuracy(outputs.data, labels.data, topk=(1, 5))
            val_acc_log.update(prec1, images.size(0))

        #logger.append([learning_rate, train_loss_log.avg, val_loss_log.avg, train_acc_log.avg, val_acc_log.avg])
        print('Accuracy of the network on the 10000 test images: %.8f %%' % (val_acc_log.avg))
        print('Loss of the network on the 10000 test images: %.8f' % (val_loss_log.avg))
        training_data['val_loss'].append(val_loss_log.avg.detach().cpu().numpy())
        training_data['val_acc'].append(val_acc_log.avg.detach().cpu().numpy())
    #logger.close()
    #logger.plot()
    training_data['learning_rate'] = learning_rate
    return training_data


'Algorithms that can be choosed'
algorithm_labels = ['0.Adam', '1.SGD', '2.adabound', '3.lb_adam', '4.lb_SGD',
                    '5.direction_Adam',  '6.direction_lbAdam', '7.direction_adamtosgd']

task = int(input('please input a task, 0 for algorithm comparing, 1 for learning rate modify, \n'))
if task == 0:
    test_algorithms = eval(input('please input testing algorithms, only list consist of int(algorithm sign) supported\n'))
    test_algorithms = [int(i) for i in test_algorithms]
    learning_rates = eval(input('please input learning rates, must corresponding to the algorithms \n'))
    if len(test_algorithms) < 1 or len(test_algorithms) != len(learning_rates):
        raise ValueError('lr and algorithms are not corresponding')
elif task == 1:
    test_algorithm = int(input('please input a single algorithm symbol \n'))
    learning_rates = eval(input('please input testing learning rates,only list supported \n'))
else:
    raise ValueError('not correct task symbol')
repeats = int(input('please input how many times to repeat \n'))

show_symbol = eval(input('please choose what to show, 0 for accuracy, 1 for loss, 2 for training_err,'
                         ' 3 for test loss,4 for test acc. support multiple chioce. please input an list \n'))
for i in show_symbol:
    i = int(i)
    if not(i == 0 or i == 1 or i == 2 or i == 3 or i == 4):
        raise  ValueError('incorrect show symbol')

shows = ['acc', 'loss', 'training_err', 'test_acc', 'test_loss']
models = ['VGG11', 'CNN', 'ResNet']
comparing_datas = [[] for i in show_symbol]
comparing_data = [[] for i in show_symbol]
test_algorithm_labels = [[] for i in show_symbol]
if task == 0:
    for i in range(len(test_algorithms)):
        for j in range(repeats):
            output = training(model_sign=model_sign, optimizer_sign=test_algorithms[i],
                              learning_rate=learning_rates[i])
            for a in range(len(show_symbol)):
                if j == 0:
                    if show_symbol[a] == 0:
                        comparing_data[a] = np.array(output['train_acc'])
                    elif show_symbol[a] == 1:
                        comparing_data[a] = np.array(output['train_loss'])
                    elif show_symbol[a] == 2:
                        comparing_data[a] = 100 - np.array(output['train_acc'])
                    elif show_symbol[a] == 3:
                        comparing_data[a] = np.array(output['val_acc'])
                    else:
                        comparing_data[a] = np.array(output['val_loss'])
                else:
                    if show_symbol[a] == 0:
                        comparing_data[a] += np.array(output['train_acc'])
                    elif show_symbol[a] == 1:
                        comparing_data[a] += np.array(output['train_loss'])
                    elif show_symbol[a] == 2:
                        comparing_data[a] += 100 - np.array(output['train_acc'])
                    elif show_symbol[a] == 3:
                        comparing_data[a] += np.array(output['val_acc'])
                    else:
                        comparing_data[a] += np.array(output['val_loss'])
        for a in range(len(show_symbol)):
            comparing_datas[a].append(np.array(comparing_data[a]) / repeats)
            test_algorithm_labels[a].append(
                algorithm_labels[test_algorithms[i]] + ' learning_rate=' + str(learning_rates[i]))
elif task == 1:
    for i in range(len(learning_rates)):
        for j in range(repeats):
            output = training(model_sign=model_sign, optimizer_sign=test_algorithm,
                              learning_rate=learning_rates[i])
            for a in range(len(show_symbol)):
                if j == 0:
                    if show_symbol[a] == 0:
                        comparing_data[a] = np.array(output['train_acc'])
                    elif show_symbol[a] == 1:
                        comparing_data[a] = np.array(output['train_loss'])
                    elif show_symbol[a] == 2:
                        comparing_data[a] = 100 - np.array(output['train_acc'])
                    elif show_symbol[a] == 3:
                        comparing_data[a] = np.array(output['val_acc'])
                    else:
                        comparing_data[a] = np.array(output['val_loss'])
                else:
                    if show_symbol[a] == 0:
                        comparing_data[a] += np.array(output['train_acc'])
                    elif show_symbol[a] == 1:
                        comparing_data[a] += np.array(output['train_loss'])
                    elif show_symbol[a] == 2:
                        comparing_data[a] += 100 - np.array(output['train_acc'])
                    elif show_symbol[a] == 3:
                        comparing_data[a] += np.array(output['val_acc'])
                    else:
                        comparing_data[a] += np.array(output['val_loss'])
        for a in range(len(show_symbol)):
            comparing_datas[a].append(np.array(comparing_data[a]) / repeats)
            test_algorithm_labels[a].append(
                algorithm_labels[test_algorithm] + ' learning_rate=' + str(learning_rates[i]))
'''
elif task == 2:
    for i in range(len(learning_rates)):
        for j in range(repeats):
            output = training(model_sign=model_sign, optimizer_sign=test_algorithm,
                              learning_rate=learning_rates[i],
                              alpha=alphas[i])
            for a in range(len(show_symbol)):
                if j == 0:
                    if show_symbol[a] == 0:
                        comparing_data[a] = np.array(output['train_acc'])
                    elif show_symbol[a] == 1:
                        comparing_data[a] = np.array(output['train_loss'])
                    elif show_symbol[a] == 2:
                        comparing_data[a] = 100 - np.array(output['train_acc'])
                    elif show_symbol[a] == 3:
                        comparing_data[a] = np.array(output['val_acc'])
                    else:
                        comparing_data[a] = np.array(output['val_loss'])
                else:
                    if show_symbol[a] == 0:
                        comparing_data[a] += np.array(output['train_acc'])
                    elif show_symbol[a] == 1:
                        comparing_data[a] += np.array(output['train_loss'])
                    elif show_symbol[a] == 2:
                        comparing_data[a] += 100 - np.array(output['train_acc'])
                    elif show_symbol[a] == 3:
                        comparing_data[a] += np.array(output['val_acc'])
                    else:
                        comparing_data[a] += np.array(output['val_loss'])
        for a in range(len(show_symbol)):
            comparing_datas[a].append(np.array(comparing_data[a]) / repeats)
            test_algorithm_labels[a].append(
                algorithm_labels[test_algorithm] + ' learning_rate=' + str(learning_rates[i]) + ' alpha=' + str(alphas[i]))
'''
save_sign = 10
with open('data/expresults/expdatas/expdata.txt', 'a') as f:
    f.write('\n\nmnist\ntraining epochs:' + str(num_epochs) + 'model_sign:' + str(model_sign))
    for a in range(len(show_symbol)):
        f.write('\nshow_symbol:' + str(a))
        for i in range(len(comparing_datas[a])):
            f.write('\n' + str(test_algorithm_labels[a][i]))
            f.write('\n' + str(comparing_datas[a][i]))
            plt.plot(range(len(comparing_datas[a][i])), comparing_datas[a][i])
        plt.legend(test_algorithm_labels[a])
        plt.title(models[model_sign] + ' mnist, ' + shows[show_symbol[a]])
        plt.savefig('data/matplotlib/' + str(save_sign))
        save_sign += 1
        plt.show()
        plt.cla()

a = 0




