import argparse
import collections
import numpy as np


import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F

import scipy.io as scio
from network import LeNet
from network2 import Adjacency
import torch.utils.data as Data
import matplotlib.pyplot as plt
import torchvision

import os


# def random_mask(x):
#     mask = torch.rand(x.shape)
#     mask = torch.round(mask)
#     result = x.mul(mask)
#     return result



BATCH_SIZE = 100
learning_rate = 1e-5
Epoch = 1000
mini_loss = 0.4519
mini_loss2 = 0.439

print_loss_frequency = 1
test_frequency = 1
mini_accuracy = 0.7

print("torch.cuda.is_available() = ", torch.cuda.is_available())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


'''load dataset and preprocess'''

autism_data = np.load('autism_400_matrix.npy')
control_data = np.load('control_400_matrix.npy')

print(autism_data.shape)

autism_data


train_autism = autism_data[80:]
test_autism = autism_data[:80]

train_control = control_data[80:]
test_control = control_data[:80]

train_autism = train_autism[:, np.newaxis, :, :]
train_control = train_control[:, np.newaxis, :, :]
test_autism = test_autism[:, np.newaxis, :, :]
test_control = test_control[:, np.newaxis, :, :]

print('loaded data')

train_label1 = np.array([[0, 1]])
train_label1 = np.repeat(train_label1, train_autism.shape[0], 0)

train_label2 = np.array([[1, 0]])
train_label2 = np.repeat(train_label2, train_control.shape[0], 0)

test_label1 = np.array([[0, 1]])
test_label1 = np.repeat(test_label1, test_autism.shape[0], 0)

test_label2 = np.array([[1, 0]])
test_label2 = np.repeat(test_label2, test_control.shape[0], 0)

train_input = np.concatenate((train_autism, train_control), axis=0)
train_output = np.concatenate((train_label1, train_label2), axis=0)

test_input = np.concatenate((test_autism, test_control), axis=0)
test_output = np.concatenate((test_label1, test_label2), axis=0)

train_input = torch.from_numpy(train_input)
train_output = torch.from_numpy(train_output)
test_input = torch.from_numpy(test_input)
test_output = torch.from_numpy(test_output)


train_torch_dataset = Data.TensorDataset(train_input, train_output)
train_loader = Data.DataLoader(
    dataset=train_torch_dataset,      # torch TensorDataset format
    batch_size=BATCH_SIZE,      # mini batch size
    shuffle=True,               # 要不要打乱数据 (打乱比较好)
)

test_torch_dataset = Data.TensorDataset(test_input, test_output)
test_loader = Data.DataLoader(
    dataset=test_torch_dataset,      # torch TensorDataset format
    batch_size=BATCH_SIZE,      # mini batch size
    shuffle=True,               # 要不要打乱数据 (打乱比较好)
)

print('processed data')


''' Model '''

model = LeNet()
adjacency_matrix = Adjacency()

model.to(device)
adjacency_matrix.to(device)

optimizer1 = optim.Adam(model.parameters(), lr=learning_rate)
optimizer2 = optim.Adam(adjacency_matrix.parameters(), lr=learning_rate)

loss_fn = nn.MSELoss(reduction='mean')

# print('load model 3 ?')

if os.path.exists('checkpoint/mask.pkl'):
    print('load model')
    adjacency_matrix.load_state_dict(torch.load('checkpoint/mask.pkl'))

if os.path.exists('checkpoint/model.pkl'):
    print('load model')
    model.load_state_dict(torch.load('checkpoint/model.pkl'))



print('start training')

for epoch in range(Epoch):

    train_acc = 0
    train_loss = 0

    for step, (train_input, train_output) in enumerate(train_loader):


        train_input = train_input.float().to(device)
        train_output = train_output.float().to(device)
        identity = torch.eye(400).float().to(device)

        optimizer1.zero_grad()
        optimizer2.zero_grad()

        mask_train_input, mask_matrix = adjacency_matrix(train_input, identity)
        train_preds = model(train_input)

        # print('train_preds: ', train_preds.shape)
        # print('train_output: ', train_output.shape)

        train_preds = torch.squeeze(train_preds)


        train_loss = loss_fn(train_preds, train_output)
        train_loss += train_loss.item()

        train_loss.backward()
        optimizer1.step()
        optimizer2.step()

    if epoch % print_loss_frequency == 0:
        # print('------------  epoch ', epoch, '  ------------')
        print('train loss: ', train_loss.item())

        predict_result = torch.max(train_preds, 1)[1]
        gt_result = torch.max(train_output, 1)[1]
        predict_result = predict_result.cpu().detach().numpy()
        gt_result = gt_result.cpu().detach().numpy()
        correct_num = np.sum(gt_result == predict_result)

        # predict_result = train_preds.round()
        # predict_result = predict_result.cpu().detach().numpy()
        # gt_result = train_output.cpu().detach().numpy()
        # correct_num = np.sum(gt_result == predict_result)

        # print('train accuracy: ', correct_num / len(gt_result))
        train_accuracy = correct_num / len(gt_result)


    if epoch % test_frequency == 0:

        test_step_num = 0
        total_accuracy = 0
        average_test_accuracy = 0

        for step, (test_input, test_output) in enumerate(test_loader):

            # print('test_input: ', test_input)

            test_step_num += 1

            test_input = test_input.float().to(device)
            test_output = test_output.float().to(device)
            identity = torch.eye(400).float().to(device)

            mask_test_input, mask_matrix = adjacency_matrix(test_input, identity)
            test_preds = model(test_input)
            test_preds = torch.squeeze(test_preds)

            # print('test: ', test_preds)

            predict_result = torch.max(test_preds, 1)[1]
            gt_result = torch.max(test_output, 1)[1]
            predict_result = predict_result.cpu().detach().numpy()
            gt_result = gt_result.cpu().detach().numpy()
            correct_num = np.sum(gt_result == predict_result)

            # mask_matrix = mask_matrix.cpu().detach().numpy()
            # plt.imshow(mask_matrix, cmap='gray')
            # plt.show()

            # predict_result = test_preds.round()
            # predict_result = predict_result.cpu().detach().numpy()
            # gt_result = test_output.cpu().detach().numpy()
            # correct_num = np.sum(gt_result == predict_result)
            #
            total_accuracy += correct_num / len(gt_result)

        average_test_accuracy = total_accuracy / test_step_num

        # print('--------------test accuracy: ', average_test_accuracy)

        # if average_test_accuracy > mini_accuracy:
        #     print('save model')
        #     torch.save(adjacency_matrix.state_dict(), 'checkpoint/mask.pkl')
        #     torch.save(model.state_dict(), 'checkpoint/model.pkl')
        #     mini_accuracy = average_test_accuracy

        if mini_loss > train_loss.item():
            print('save model')
            # torch.save(adjacency_matrix.state_dict(), 'checkpoint/mask.pkl')
            torch.save(model.state_dict(), 'checkpoint/model.pkl')
            mini_loss = train_loss.item()

        # if mini_loss2 > train_loss.item():
        #     print('save model')
        #     torch.save(adjacency_matrix.state_dict(), 'checkpoint/mask.pkl')
        #     # torch.save(model.state_dict(), 'checkpoint/model.pkl')
        #     mini_loss2 = train_loss.item()