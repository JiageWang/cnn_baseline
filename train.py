import time

import torch

from utils import AverageMeter, ProgressMeter, accuracy


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    # top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1],  # , top5],
        prefix="Epoch: [{}]".format(epoch))

    accumulate_acc1 = 0
    accumulate_loss = 0
    accumulate_count = 0
    steps = args.accumulate_steps

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

        output = model(images)
        loss = criterion(output, target)
        acc1 = accuracy(output, target, topk=(1,))[0]
        accumulate_acc1 += acc1.item()
        accumulate_loss += loss.item()
        accumulate_count += images.size(0)

        loss.backward()
        if (i + 1) % steps == 0:
            losses.update(accumulate_loss / steps, accumulate_count)
            top1.update(accumulate_acc1 / steps, accumulate_count)
            accumulate_loss = 0
            accumulate_acc1 = 0
            accumulate_count = 0

            optimizer.step()
            optimizer.zero_grad()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
    return top1.avg, losses.avg