# -*- coding:utf-8 -*-
import argparse
import os
import random
import datetime
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import Net, L1_Charbonnier_loss, L1_Sobel_loss
from dataset import DatasetFromImage
from test import eval

parser = argparse.ArgumentParser(description="Super Resolution Trainer")
parser.add_argument("--batchSize", type=int, default=32,
                    help="training batch size")
parser.add_argument("--nEpochs", type=int, default=1000,
                    help="number of epochs to train for")
parser.add_argument("--lr", type=float, default=1e-4,
                    help="Learning Rate. Default=1e-4")
parser.add_argument("--step", type=int, default=300,
                    help="Sets the learning rate to the initial LR decayed by momentum every n epochs, Default: n=10")
parser.add_argument("--resume", default="", type=str,
                    help="Path to checkpoint (default: none)")
parser.add_argument("--start-epoch", default=1, type=int,
                    help="Manual epoch number (useful on restarts)")
parser.add_argument("--threads", type=int, default=8,
                    help="Number of threads for data loader to use, Default: 1")
parser.add_argument("--pretrained", default="./checkpoints/model_best_du.pth",
                    type=str, help="path to pretrained model (default: none)")


def adjust_learning_rate(epoch):
    lr = opt.lr * (0.1 ** (epoch // opt.step))
    return lr


def train(training_data_loader, optimizer, model, criterion, criterion_sobel, epoch):
    lr = adjust_learning_rate(epoch-1)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    print("epoch =", epoch, "lr =", optimizer.param_groups[0]["lr"])
    model.train()     

    for iteration, batch in enumerate(training_data_loader, 1):
        input = batch[0]   # LR input
        label = batch[1]   # HR label
        input = input.cuda()
        label = label.cuda()

        res = model(input)
        loss1 = criterion(res, label)
        loss2 = criterion_sobel(res, label)
        loss = 0.9*loss1+0.1*loss2
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if iteration % 20 == 0:
            print("===> Epoch[{}]({}/{}): Loss: {:.10f}".format(epoch,
                  iteration, len(training_data_loader), loss.item()))
    return model


def save_checkpoint(model, epoch):
    model_folder = "checkpoints/"
    model_out_path = model_folder + "model_best_du.pth"
    state = {"epoch": epoch, "model": model}
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    torch.save(state, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))


if __name__ == "__main__":
    global opt, model
    opt = parser.parse_args()
    print(opt)
    opt.seed = random.randint(1, 10000)
    print("Random Seed: ", opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    cudnn.benchmark = True

    print("===> Loading datasets")
    train_set = DatasetFromImage("/home/chengxi/datasets/DIV2K_train_HR")
    training_data_loader = DataLoader(
        dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)

    print("===> Building model")
    model = Net()
    criterion = L1_Charbonnier_loss()
    criterion_sobel = L1_Sobel_loss()

    print("===> Setting GPU")
    model = nn.DataParallel(model, device_ids=[0, 1]).cuda()
    criterion = criterion.cuda()
    criterion_sobel = criterion_sobel.cuda()

    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            opt.start_epoch = checkpoint["epoch"] + 1
            saved_state = checkpoint["model"].state_dict()
            model.load_state_dict(saved_state)
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))

    if opt.pretrained:
        if os.path.isfile(opt.pretrained):
            print("=> loading model '{}'".format(opt.pretrained))
            weights = torch.load(opt.pretrained)
            pretrained_dict = weights['model'].state_dict()
            model_dict = model.state_dict()
            pretrained_dict = {k: v for k,
                               v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
        else:
            print("=> no model found at '{}'".format(opt.pretrained))

    print("===> Setting Optimizer")
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)

    print("===> Training")
    best_psnr = 0
    start_time = datetime.datetime.now()
    for epoch in range(opt.start_epoch, opt.nEpochs + 1):
        model = train(training_data_loader, optimizer, model,
                      criterion, criterion_sobel, epoch)
        model.eval()
        psnr, ssim = eval(testdir='Set5', model=model)
        if psnr > best_psnr:
            best_psnr = psnr
            save_checkpoint(model, epoch)
        print("Best PSNR: {:.4f}, Current PSNR: {:.4f}".format(
            best_psnr, psnr))
    endtime = datetime.datetime.now()
    print("time used:", (endtime-start_time).seconds)
