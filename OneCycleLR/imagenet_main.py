import argparse
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from tqdm import tqdm, trange

from ExpUtils import init_debug, form_dir_path, wlog, init_logger_board, set_file_logger, logger, vlog


model_names = sorted(name for name in models.__dict__ if not name.startswith("__") and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', metavar='DIR', help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50', choices=model_names, help='model architecture (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=20, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int, metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--optimizer', default='sgd', type=str, help='optimizer sgd or adam, adam uses 0.001 (default: sgd)')
parser.add_argument('--lr', '--learning-rate', default=0.05, type=float, metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--blr', default=0.05, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--elr', default=0.00005, type=float, metavar='LR', help='final learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument("--decay_epochs", nargs="+", type=int, default=[60, 75, 90], help="decay learning rate by decay_rate at these epochs")
parser.add_argument('--wd', '--weight-decay', default=3e-6, type=float, metavar='W', help='weight decay (default: 1e-4)', dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int, metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true', help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int, help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int, help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str, help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str, help='distributed backend')
parser.add_argument('--seed', default=None, type=int, help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('-depth', default=28, type=int, metavar='N', help='depth for WideResNet (default: 28)')
parser.add_argument('--k', default=2, type=float, metavar='N', help='wide factor rate for WideResNet, 0.5, 1, 2 ... 10 (default: 2)')
parser.add_argument('-drop', default=0.0, type=float, metavar='N', help='dropout rate for WideResNet (default: 0.3)')
parser.add_argument("--norm", type=str, default='batch', choices=[None, 'none', "batch", "instance", "layer", "act"], help="norm to add to weights, none works fine")

parser.add_argument("--save_dir", type=str, default='./experiment')
parser.add_argument("--dir_path", type=str, default='./experiment')
parser.add_argument("--log_dir", type=str, default='./runs')
parser.add_argument("--log_arg", type=str, default='arch-epochs-optimizer-lr-weight_decay-batch_size')
parser.add_argument("--vis", action="store_true", help="")

cur_iter = 0
best_acc1 = 0
print = wlog


def main():
    args = parser.parse_args()
    args.norm = None if args.norm == 'none' else args.norm
    args.dataset = args.data.split('/')[2]
    init_debug(args)
    args.dir_path = form_dir_path('SCE', args)
    set_file_logger(logger, args)
    args.save_dir = args.dir_path
    print("save log to %s" % args.dir_path)
    init_logger_board(args)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)
    print("save log to %s" % args.dir_path)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    # create model
    try:
        size = int(args.data.split('/')[2].replace('imagenet', '').replace('png', ''))
    except:
        size = 256

    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True, num_classes=1000, size=size)
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch](num_classes=1000, size=size)

    cudnn.benchmark = True
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.482, 0.458, 0.408],
                                     std=[0.229, 0.261, 0.276])
    # normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
    #                                  std=[0.5, 0.5, 0.5])

    crop_size = 224
    if '32' in args.data:
        crop_size = 32
    elif '128' in args.data:
        crop_size = 128
    elif '64' in args.data:
        crop_size = 64

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(size),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            normalize
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    else:
        # 0.0001 for wrn,  TODO:  resnet use 0.001 or 0.0001?
        optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, steps_per_epoch=len(train_loader), epochs=args.epochs,
                                                    anneal_strategy='linear', div_factor=int(args.lr / args.blr),
                                                    final_div_factor=int(args.lr / args.elr))
    begin = time.time()
    for epoch in trange(args.start_epoch, args.epochs):

        epoch_begin = time.time()
        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, scheduler, args)

        # evaluate on validation set
        loss, acc1, acc5 = validate(val_loader, model, criterion, args)
        epoch_end = time.time()
        if args.vis:
            vlog(args.writer, epoch, 'Test', wlog, verbose=False, loss=loss, acc1=acc1, acc5=acc5)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        print("%d epoch %d, time %.2f" % (os.getpid(), epoch, epoch_end - epoch_begin))

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                    and args.rank % ngpus_per_node == 0):
            fname = "%s/%s.pth.tar" % (args.dir_path, args.arch)
            if args.arch == "wrn":
                fname = "%s-%d-%s.pth.tar" % (args.arch, args.depth, str(args.k))
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer': optimizer.state_dict(),
            }, is_best, filename=fname)
    print("num epoch %d, time %.2f" % (args.epochs, epoch_end - begin))


def train(train_loader, model, criterion, optimizer, epoch, scheduler, args):
    global cur_iter
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    loader = tqdm(train_loader)

    for i, (images, target) in enumerate(loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        if args.arch == 'wrn':
            output = model(images, logits=True)
        else:
            output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        # if i % args.print_freq == 0:
        #     progress.display(i)
        loader.set_postfix(loss='%.4f' % loss.item(), acc1='%.4f' % acc1.item(), acc5='%.4f' % acc5.item())
        cur_iter += 1
        if args.vis and cur_iter % 100 == 0:
            vlog(args.writer, cur_iter, 'Train', loss=loss.item(), acc1=acc1.item(), acc5=acc5.item())
        scheduler.step()


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            if args.arch == 'wrn':
                output = model(images, logits=True)
            else:
                output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            # if i % args.print_freq == 0:
            #     progress.display(i)

        # # TODO: this should also be done with the ProgressMeter

    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))
    return losses.avg, top1.avg, top5.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every args.decay_step epochs"""
    lr = args.lr * (0.1 ** (epoch // args.decay_step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
