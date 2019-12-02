import os
import argparse

import torch
from torch import nn
import pretrainedmodels
from tensorboardX import SummaryWriter

from dataset import get_data_loader
from train import train
from valid import valid

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--arch', default='se_resnet50', type=str)
parser.add_argument('--epochs', default=200, type=int)
parser.add_argument('--input_size', default=331, type=int)
parser.add_argument('--num_classes', default=54, type=int)
parser.add_argument('--batch_size', default=4, type=int)
parser.add_argument('--accumulate_steps', default=8, type=int)
parser.add_argument('--optimizer', default='SGD', type=str)
parser.add_argument('--learning_rate', default=0.0001, type=float)
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--weight_decay', default=1e-4, type=float)
parser.add_argument('--print_freq', default=100, type=int)
parser.add_argument('--valid_freq', default=1, type=int)
parser.add_argument('--save_acc_thred', default=95, type=float)
parser.add_argument('--resume', default='', type=str, metavar='PATH')
parser.add_argument('--train_path', default=r'C:\Users\Administrator\Desktop\train_val\train', type=str)
parser.add_argument('--valid_path', default=r'C:\Users\Administrator\Desktop\train_val\val', type=str)
parser.add_argument('--finetune', default='adaptiveavgpool2d', type=str)

best_acc1 = 0
start_epoch = 0


def main():
    global best_acc1, start_epoch
    args, unknown = parser.parse_known_args()

    # 创建
    model = pretrainedmodels.__dict__[args.arch](num_classes=1000, pretrained='imagenet')
    model.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
    num_ftrs = model.last_linear.in_features
    model.last_linear = nn.Linear(num_ftrs, args.num_classes)

    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model).cuda()
    print(model)

    # 数据集
    train_loader, valid_loader = get_data_loader(args)

    # 损失函数与优化器
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), args.learning_rate,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # 实验记录存储
    save_dir = "{}_batch{}_size{}_optim{}_lr{}_{}".format(args.arch,
                                                          args.batch_size * args.accumulate_steps,
                                                          args.input_size,
                                                          args.optimizer,
                                                          args.learning_rate,
                                                          args.finetune)
    save_log_dir = os.path.join("logs", save_dir)
    save_model_dir = os.path.join("models", save_dir)
    if not os.path.exists(save_log_dir):
        os.makedirs(save_log_dir)
    if not os.path.exists(save_model_dir):
        os.makedirs(save_model_dir)
    writer = SummaryWriter(save_log_dir)

    if args.resume:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['state_dict'])
        print("resumed best valid acc: ", checkpoint['best_acc1'])
        start_epoch = checkpoint['epoch'] + 1

    for epoch in range(start_epoch, args.epochs):
        # 训练
        train_acc1, train_loss = train(train_loader, model, criterion, optimizer, epoch, args)
        # 验证
        if (epoch + 1) % args.valid_freq == 0:
            valid_acc1 = valid(valid_loader, model, criterion, args)
            if valid_acc1 >= best_acc1 and valid_acc1 > args.save_acc_thred:
                torch.save({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'best_acc1': best_acc1,
                    'optimizer': optimizer.state_dict(),
                }, "{}/{}@epoch{}_acc{:.2f}".format(save_model_dir, args.arch, epoch, valid_acc1))
                best_acc1 = valid_acc1
            writer.add_scalar('valid_acc', valid_acc1, epoch)
        writer.add_scalar('loss', train_loss, epoch)
        writer.add_scalar('train_acc', train_acc1, epoch)
        writer.add_histogram('param/weights', model.module.last_linear.weight.data.cpu().numpy(), epoch)
        writer.add_histogram('param/bias', model.module.last_linear.bias.data.cpu().numpy(), epoch)


if __name__ == "__main__":
    main()
