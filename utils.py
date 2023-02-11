import math, random
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.distributions.normal import Normal
import matplotlib.pyplot as plt
from torchvision.datasets import MNIST
from torch.utils.data import TensorDataset, Dataset, DataLoader
import matplotlib as plt
from common import AverageMeter
from FCN import *
import os
import fgsm
import pgd
import cw
from CBDNet import *
from MLP import *
from CausalIB import *
os.environ["GIT_PYTHON_REFRESH"] = "quiet"
from scipy.stats import entropy



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

batch_size = 100
samples_amount = 12


"                                                     "
"               Daten Vorbereitung                    "
"                                                     "

train_dataset = MNIST('./data', download=True, train=True, transform=transforms.ToTensor())

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


test_dataset = MNIST('./data', download=True, train=False, transform=transforms.ToTensor())

test_loader = DataLoader(test_dataset, batch_size=batch_size)


"                                                     "
"               Modell Speichung                      "
"                                                     "

def save(net, name):
    path = './model'
    if not os.path.exists(path):
        os.mkdir(path)
    net_path = path + '/' + name +'.pkl'
    net = net.cpu()
    torch.save(net.state_dict(), net_path)
    net.to(device)

def load(net, name):
    net_path = './model/' + name +'.pkl'
    net.load_state_dict(torch.load(net_path))
    net.to(device)
    return net

def count_entropy(input):
    num = len(input)
    input = input.cpu().detach().numpy()
    # count_array = Counter(pixels).values()
    total_loss = 0
    for i in input:
        total_loss += entropy(i)

    return total_loss / num

def count_distance(clean, adver):
    num = len(clean)
    return torch.sum((clean-adver) ** 2)
"                                                     "
"               Modell Trainierung                    "
"                                                     "

def MLP_Train(model, num_epoch):

    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(num_epoch):
        loss_bei_epoch = []
        accuracy_bei_epoch = []

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            y_pre = model(x_batch)
            loss = model.batch_loss(y_pre, y_batch)
            loss_bei_epoch.append(loss.item())

            y_prediction = torch.max(y_pre, dim=1)[1]
            accuracy = torch.mean((y_prediction == y_batch).float())
            accuracy_bei_epoch.append(accuracy.item())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # if(epoch%5 == 0):
        print("EPOCH: ", epoch, ", loss: ", np.mean(loss_bei_epoch), ", Accuracy: ", np.mean(accuracy_bei_epoch))
    save(model, "MLP")

def MLP_Eval(model, name):
    model = load(model, name)
    model.eval()
    accuracy_ = []
    loss_ = []
    I_X_T_ = []
    I_Y_T_ = []

    for x_batch, y_batch in test_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        y_pre = model(x_batch)
        loss = model.batch_loss(x_batch, y_batch)
        loss_.append(loss.item())


        y_prediction = torch.max(y_pre, dim=1)[1]
        accuracy = torch.mean((y_prediction == y_batch).float())
        accuracy_.append(accuracy.item())


    # if(epoch%5 == 0):
    print("TEST, Accuracy: ", np.mean(accuracy_), "loss: ",  np.mean(loss_))

def MLP_AT_Train(model, num_epoch, epsilon, name, adver_type):
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    if adver_type == "fgsm":
      adver_image_obtain = fgsm.attack_model(model=model)
    elif adver_type == "pgd":
      adver_image_obtain = pgd.attack_model(model=model)
    for epoch in range(num_epoch):
        loss_bei_epoch = []
        accuracy_bei_epoch = []

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            y_pre = model(x_batch)
            loss = model.batch_loss(y_pre, y_batch)
            loss_bei_epoch.append(loss.item())

            y_prediction = torch.max(y_pre, dim=1)[1]
            accuracy = torch.mean((y_prediction == y_batch).float())
            accuracy_bei_epoch.append(accuracy.item())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            perturbed_x_batch = adver_image_obtain.generate(x_batch, eps=epsilon, y=y_batch)

            y_pre = model(perturbed_x_batch)
            loss = model.batch_loss(y_pre, y_batch)
            loss_bei_epoch.append(loss.item())

            y_prediction = torch.max(y_pre, dim=1)[1]
            accuracy = torch.mean((y_prediction == y_batch).float())
            accuracy_bei_epoch.append(accuracy.item())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # if(epoch%5 == 0):
        print("EPOCH: ", epoch, ", loss: ", np.mean(loss_bei_epoch), ", Accuracy: ", np.mean(accuracy_bei_epoch))
    save(model, name)

def InfoAT_Train(model, num_epoch, epsilon, name, adver_type):
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    if adver_type == "fgsm":
      adver_image_obtain = fgsm.attack_model(model=model)
    elif adver_type == "pgd":
      adver_image_obtain = pgd.attack_model(model=model)
    for epoch in range(num_epoch):
        loss_bei_epoch = []
        accuracy_bei_epoch = []

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            y_pre = model(x_batch)
            clean_entropy_loss = count_entropy(y_pre)
            # loss = model.batch_loss(y_pre, y_batch)
            # loss_bei_epoch.append(loss.item())

            y_prediction = torch.max(y_pre, dim=1)[1]
            accuracy = torch.mean((y_prediction == y_batch).float())
            accuracy_bei_epoch.append(accuracy.item())

            perturbed_x_batch = adver_image_obtain.generate(x_batch, eps=epsilon, y=y_batch)

            y_pre_hat = model(perturbed_x_batch)
            adver_entropy_loss = count_entropy(y_pre_hat)
            CE_loss = model.batch_loss(y_pre_hat, y_batch)

            distance = count_distance(y_pre, y_pre_hat)

            y_prediction = torch.max(y_pre_hat, dim=1)[1]
            accuracy = torch.mean((y_prediction == y_batch).float())
            accuracy_bei_epoch.append(accuracy.item())

            loss = CE_loss - 0.5 * adver_entropy_loss + 1 * clean_entropy_loss * distance
            loss_bei_epoch.append(loss.item())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # if(epoch%5 == 0):
        print("EPOCH: ", epoch, ", loss: ", np.mean(loss_bei_epoch), ", Accuracy: ", np.mean(accuracy_bei_epoch))
    save(model, name)

def MLP_Adver(model, epsilon, name, adver_type):
    model = load(model, name)

    model.eval()
    if adver_type == "fgsm":
        adver_image_obtain = fgsm.attack_model(model=model)
    elif adver_type == "pgd":
        adver_image_obtain = pgd.attack_model(model=model)
    elif adver_type == "cw":
        adver_image_obtain = cw.attack_model(model=model)
    else:
        print("False Type")
    accuracy_clean = []
    accuracy_adver = []

    for x_batch, y_batch in test_loader:

        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        y_pre = model(x_batch)
        y_prediction = torch.max(y_pre, dim=1)[1]
        accuracy = torch.mean((y_prediction == y_batch).float())
        accuracy_clean.append(accuracy.item())


        perturbed_x_batch = adver_image_obtain.generate(x_batch, eps=epsilon, y=y_batch)


        y_pre = model(perturbed_x_batch)
        y_prediction = torch.max(y_pre, dim=1)[1]
        accuracy = torch.mean((y_prediction == y_batch).float())
        accuracy_adver.append(accuracy.item())


    # if(epoch%5 == 0):
    print("TEST, Clean Accuracy: ", np.mean(accuracy_clean), ", Adversial Accuracy: ",  np.mean(accuracy_adver))

def CBDNet_Train(CBDNet, model, num_epoch, epsilon, name_CBDNet, name_MLP, load_model=False, load_name="name"):
    losses = AverageMeter()
    if load_model:
      model = load(model, load_name)
    model.train()
    criterion = fixed_loss().to(device)
    optimizer_CBDNet = optim.Adam(model.parameters(), lr=1e-4)
    optimizer_MLP = optim.Adam(model.parameters(), lr=1e-4)
    adver_image_obtain = fgsm.attack_model(model=model)
    for epoch in range(num_epoch):
        CBD_loss_ = []
        MLP_loss = []
        clean_accuracy = []
        adver_accuracy = []
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            # MLP Train
            y_pre = model(x_batch)
            clean_loss = model.batch_loss(y_pre, y_batch)

            y_prediction = torch.max(y_pre, dim=1)[1]
            accuracy = torch.mean((y_prediction == y_batch).float())
            clean_accuracy.append(accuracy.item())

            optimizer_MLP.zero_grad()
            clean_loss.backward()
            # optimizer_MLP.step()

            # Generate Adver Examples
            perturbed_x_batch = adver_image_obtain.generate(x_batch, eps=epsilon, y=y_batch)

            # Train CBDNet
            input_var = perturbed_x_batch
            target_var = x_batch
            sigma_var = input_var - target_var

            noise_level_est, output = CBDNet(input_var)

            CBD_loss = criterion(output, target_var, noise_level_est, sigma_var)
            # loss = criterion(output, target_var, noise_level_est, sigma_var, flag_var)
            losses.update(CBD_loss.item())
            CBD_loss_.append(CBD_loss.item())
            optimizer_CBDNet.zero_grad()
            CBD_loss.backward()
            optimizer_CBDNet.step()

            # Train AT MLP
            y_pre = model(perturbed_x_batch)
            adver_loss = model.batch_loss(y_pre, y_batch)
            MLP_loss.append(adver_loss.item()+clean_loss.item())

            y_prediction = torch.max(y_pre, dim=1)[1]
            accuracy = torch.mean((y_prediction == y_batch).float())
            adver_accuracy.append(accuracy.item())

            adver_loss.backward()
            optimizer_MLP.step()
            # optimizer_MLP.zero_grad()
        print("EPOCH: ", epoch, ", CBD loss: ", np.mean(CBD_loss_),", MLP loss: ", np.mean(MLP_loss), ", clean accuracy: ", np.mean(clean_accuracy), ", adver accuracy: ", np.mean(adver_accuracy))
    save(model, name_MLP)
    save(CBDNet, name_CBDNet)

def CBD_Adver(CBDNet, model, epsilon, MLP_name, CBD_name, adver_type):
    model = load(model, MLP_name)
    CBDNet = load(CBDNet, CBD_name)
    model.eval()
    if adver_type == "fgsm":
        adver_image_obtain = fgsm.attack_model(model=model)
    elif adver_type == "pgd":
        adver_image_obtain = pgd.attack_model(model=model)
    elif adver_type == "cw":
        adver_image_obtain = cw.attack_model(model=model)
    else:
        print("False Type")
    accuracy_clean = []
    accuracy_adver = []

    for x_batch, y_batch in test_loader:

        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        y_pre = model(x_batch)
        y_prediction = torch.max(y_pre, dim=1)[1]
        accuracy = torch.mean((y_prediction == y_batch).float())
        accuracy_clean.append(accuracy.item())


        perturbed_x_batch = adver_image_obtain.generate(x_batch, eps=epsilon, y=y_batch)

        _, output = CBDNet(perturbed_x_batch)
        y_pre = model(output)
        y_prediction = torch.max(y_pre, dim=1)[1]
        accuracy = torch.mean((y_prediction == y_batch).float())
        accuracy_adver.append(accuracy.item())


    # if(epoch%5 == 0):
    print("TEST, Clean Accuracy: ", np.mean(accuracy_clean), ", Adversial Accuracy: ",  np.mean(accuracy_adver))

def causalIB_InfoAT_train(model, num_epoch, epsilon, name, adver_type):
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    if adver_type == "fgsm":
      adver_image_obtain = fgsm.attack_model(model=model)
    elif adver_type == "pgd":
      adver_image_obtain = pgd.attack_model(model=model)
    for epoch in range(num_epoch):
        loss_bei_epoch = []
        accuracy_bei_epoch = []
        clean_I_X_T_bei_epoch = []
        clean_I_Y_T_bei_epoch = []

        adver_I_X_T_bei_epoch = []
        adver_I_Y_T_bei_epoch = []

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            # clean train
            y_pre, mean, std, features, y_logitss, z_scores = model.myForward(x_batch)
            loss, I_X_T, I_Y_T = model.batch_loss(mean, std, y_logitss, features, z_scores, y_batch)
            clean_I_X_T_bei_epoch.append(I_X_T.item())
            clean_I_Y_T_bei_epoch.append(I_Y_T.item())
            clean_entropy_loss = count_entropy(y_pre)

            # calculate clean accuracy
            y_prediction = torch.max(y_pre, dim=1)[1]
            accuracy = torch.mean((y_prediction == y_batch).float())
            accuracy_bei_epoch.append(accuracy.item())

            # generate perturbed adver examples
            perturbed_x_batch = adver_image_obtain.generate(x_batch, eps=epsilon, y=y_batch)

            # adver train
            y_pre_hat, mean, std, features, y_logitss, z_scores = model.myForward(perturbed_x_batch)
            adver_entropy_loss = count_entropy(y_pre_hat)
            CE_loss, I_X_T, I_Y_T = model.batch_loss(mean, std, y_logitss, features, z_scores, y_batch)
            adver_I_X_T_bei_epoch.append(I_X_T.item())
            adver_I_Y_T_bei_epoch.append(I_Y_T.item())

            # calculate loss
            distance = count_distance(y_pre, y_pre_hat)

            y_prediction = torch.max(y_pre_hat, dim=1)[1]
            accuracy = torch.mean((y_prediction == y_batch).float())
            accuracy_bei_epoch.append(accuracy.item())

            loss = CE_loss - 0.5 * adver_entropy_loss + 1 * clean_entropy_loss * distance

            loss_bei_epoch.append(loss.item())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # if(epoch%5 == 0):
        print("EPOCH: ", epoch, ", loss: ", np.mean(loss_bei_epoch), ", Accuracy: ", np.mean(accuracy_bei_epoch))
    save(model, name)
