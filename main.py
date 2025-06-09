import argparse
import os
import random
import time
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import models
from tensorboardX import SummaryWriter
from sklearn.metrics import confusion_matrix
from utils import *
from imbalance_cifar import IMBALANCECIFAR10, IMBALANCECIFAR100
import wandb
from utils.utils import *
from utils.autoaug import CIFAR10Policy, Cutout

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch Cifar Training')
parser.add_argument('--dataset', default='cifar10', help='dataset setting')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet32',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet32)')
parser.add_argument('--imb_type', default="exp", type=str, help='imbalance type')
parser.add_argument('--imb_factor', default=0.01, type=float, help='imbalance factor')
parser.add_argument('--rand_number', default=0, type=int, help='fix random number for data sampling')
parser.add_argument('--exp_str', default='0', type=str, help='number to indicate which experiment it is')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--save_freq', default=10, type=int, metavar='N',
                    help='Save the checkpoints after every n epochs')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N',
                    help='mini-batch size')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                     dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--seed', default=0, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')
parser.add_argument('--root_log',type=str, default='log')
parser.add_argument('--root_model', type=str, default='checkpoint')
parser.add_argument('--log_results', action='store_true',
                    help='To log results on wandb')
parser.add_argument('--name', type=str, default='test')
parser.add_argument('--warmup', type=int, default=160)
parser.add_argument('--randaug', default=1, type=int)
parser.add_argument('--m', default=0.9, type=float, help='momentum in rdr')

best_acc1 = 0
best_acc_detail = np.zeros(3)

def main():
    args = parser.parse_args()
    
    store_dir_list = [args.dataset, args.arch, args.imb_type, str(args.imb_factor)]
    store_name_list = ['seed', str(args.seed), 'wd', '%.6f' % args.weight_decay, 'lr', str(args.lr), 'm', str(args.m)]
    store_dir = '_'.join(store_dir_list)
    args.store_name = '_'.join(store_name_list)
    args.store_name = os.path.join(store_dir, args.store_name)
    prepare_folders(args)

    if args.seed is not None:
        random.seed(args.seed)
        os.environ['PYTHONHASHSEED'] = str(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    ngpus_per_node = torch.cuda.device_count()
    main_worker(args.gpu, ngpus_per_node, args)

def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    global best_acc_detail
    args.gpu = gpu
    if args.log_results:
        wandb.init(project="long-tail-cifar", entity="user", name=args.store_name)
        wandb.config.update(args)
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    # create model
    print("=> creating model '{}'".format(args.arch))
    num_classes = 100 if args.dataset == 'cifar100' else 10
    use_norm = False
    model = models.__dict__[args.arch](num_classes=num_classes, use_norm=use_norm)

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        model = torch.nn.DataParallel(model).cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location='cuda:0')
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}'".format(args.resume))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # data augment
    if args.randaug:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            CIFAR10Policy(),    
            transforms.ToTensor(),
            Cutout(n_holes=1, length=16),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
    else:
        transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    if args.dataset == 'cifar10':
        train_dataset = IMBALANCECIFAR10(root='./data', imb_type=args.imb_type, imb_factor=args.imb_factor, rand_number=args.rand_number, train=True, download=True, transform=transform_train)
        val_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_val)
    elif args.dataset == 'cifar100':
        train_dataset = IMBALANCECIFAR100(root='./data', imb_type=args.imb_type, imb_factor=args.imb_factor, rand_number=args.rand_number, train=True, download=True, transform=transform_train)
        val_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_val)
    else:
        warnings.warn('Dataset is not listed')
        return
    
    cls_num_list = train_dataset.get_cls_num_list()
    print('cls num list:')
    print(cls_num_list)
    args.cls_num_list = cls_num_list
    
    train_sampler = None
        
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=100, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # init log for training
    log_training = open(os.path.join(args.root_log, args.store_name, 'log_train.csv'), 'w')
    log_testing = open(os.path.join(args.root_log, args.store_name, 'log_test.csv'), 'w')
    with open(os.path.join(args.root_log, args.store_name, 'args.txt'), 'w') as f:
        f.write(str(args))
    tf_writer = SummaryWriter(log_dir=os.path.join(args.root_log, args.store_name))
    
    feature_dim = 64  
    # Phi_nu ini
    Phi_nu = torch.zeros(feature_dim, num_classes, device=args.gpu) 
    updated_labels = set()

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args)

        per_cls_weights = None
        criterion = nn.CrossEntropyLoss(weight=per_cls_weights, reduction='none').cuda(args.gpu)

        if args.evaluate:
            print("=> evaluating model on validation set")
            if args.resume:
                acc1, acc_detail = validate(val_loader, model, criterion, 0, args, log_testing, tf_writer)
                print('Evaluation Acc@1: {:.3f}'.format(acc1))
                print('Evaluation head, mid, tail Accuracy: {}'.format(np.array2string(acc_detail, 
                    separator=',', formatter={'float_kind':lambda x: "%.3f" % x})))
                if log_testing:
                    log_testing.write('Evaluation Acc@1: {:.3f}\n'.format(acc1))
                    log_testing.flush()
                return

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args, log_training, tf_writer, cls_num_list, Phi_nu, updated_labels)
        
        # evaluate on validation set
        acc1, acc_detail = validate(val_loader, model, criterion, epoch, args, log_testing, tf_writer)
        
        if args.log_results:
            wandb.log({'epoch':epoch, 'val_acc':acc1})
        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        if is_best:
            best_acc_detail = acc_detail

        tf_writer.add_scalar('acc/test_top1_best', best_acc1, epoch)
        output_best = 'Best Prec@1: %.3f' % (best_acc1)
        print(output_best)
        output_best_detail = 'Best head, mid, tail Accuracy: %s\n'% (np.array2string(best_acc_detail, separator=',', formatter={'float_kind':lambda x: "%.3f" % x}))
        print(output_best_detail)
        log_testing.write(output_best + '\n')
        log_testing.write(output_best_detail + '\n')
        log_testing.flush()

        save_checkpoint(args, {
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
            'optimizer' : optimizer.state_dict(),
        }, is_best, args.save_freq)

    if args.log_results:
        wandb.log({'best_acc':best_acc1})


def train(train_loader, model, criterion, optimizer, epoch, args, log, tf_writer, cls_num_list, Phi_nu, updated_labels):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    
    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            input = input.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output, feature = model(input)        

        if epoch < args.warmup:
            sample_weights = torch.ones_like(target, dtype=torch.float, device=target.device)
        else:
            sample_weights = rdr_weights(feature, target, cls_num_list, Phi_nu, args)

        loss = criterion(output, target) * sample_weights
        loss = loss.mean()

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        update_Phi_nu(feature, target, Phi_nu, args, updated_labels)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        if args.log_results:
            wandb.log({'loss':loss, 'top1_acc':acc1, 'top5_acc':acc5})
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            output = ('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5, lr=optimizer.param_groups[-1]['lr'] * 0.1)) 
            print(output)
            log.write(output + '\n')
            log.flush()

    tf_writer.add_scalar('loss/train', losses.avg, epoch)
    tf_writer.add_scalar('acc/train_top1', top1.avg, epoch)
    tf_writer.add_scalar('acc/train_top5', top5.avg, epoch)
    tf_writer.add_scalar('lr', optimizer.param_groups[-1]['lr'], epoch)


def validate(val_loader, model, criterion, epoch, args, log=None, tf_writer=None, flag='val'):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    
    # switch to evaluate mode
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output, _ = model(input)
            loss = criterion(output, target).mean()

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            _, pred = torch.max(output, 1)
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

            if i % args.print_freq == 0:
                output = ('Test: [{0}/{1}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                          'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1, top5=top5))
                print(output)

        cf = confusion_matrix(all_targets, all_preds).astype(float)
        cls_cnt = cf.sum(axis=1)
        cls_hit = np.diag(cf)
        cls_acc = cls_hit / cls_cnt
        if args.log_results:
            if args.dataset=="cifar10":
                for idx in range(len(cls_acc)):
                    wandb.log({f"class{idx}_acc":cls_acc[idx]})
            if args.dataset=="cifar100":
                wandb.log({f"class0_acc":cls_acc[0]})
                for idx in range(1,5):
                    wandb.log({f"class{100-idx}_acc":cls_acc[-idx]})
                    wandb.log({f"class{idx}_acc":cls_acc[idx]})
        output = ('{flag} Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.5f}'
                .format(flag=flag, top1=top1, top5=top5, loss=losses))
        out_cls_acc = '%s Class Accuracy: %s'%(flag,(np.array2string(cls_acc, separator=',', formatter={'float_kind':lambda x: "%.3f" % x})))
        print(output)
        print(out_cls_acc)

        if args.dataset=="cifar10":
            cls_acc_array = np.array(cls_acc)
            acc_detail = np.array([cls_acc_array[:3].mean(), cls_acc_array[3:7].mean(), cls_acc_array[7:].mean()])
        elif args.dataset=="cifar100":
            cls_acc_array = np.array(cls_acc)
            acc_detail = np.array([cls_acc_array[:35].mean(), cls_acc_array[35:69].mean(), cls_acc_array[69:].mean()])

        if log is not None:
            log.write(output + '\n')
            log.write(out_cls_acc + '\n')
            log.flush()

        tf_writer.add_scalar('loss/test_'+ flag, losses.avg, epoch)
        tf_writer.add_scalar('acc/test_' + flag + '_top1', top1.avg, epoch)
        tf_writer.add_scalar('acc/test_' + flag + '_top5', top5.avg, epoch)
        tf_writer.add_scalars('acc/test_' + flag + '_cls_acc', {str(i):x for i, x in enumerate(cls_acc)}, epoch)

    return top1.avg, acc_detail


def update_Phi_nu(feature, target, Phi_nu, args, updated_labels):
    with torch.no_grad():
        class_feature_sum = {}
        class_sample_count = {}

        for index in range(feature.size(0)):
            label = target[index].item()
            if label in class_feature_sum:
                class_feature_sum[label] += feature[index].detach()
                class_sample_count[label] += 1
            else:
                class_feature_sum[label] = feature[index].detach()
                class_sample_count[label] = 1

        for label in class_feature_sum:
            class_mean_feature = class_feature_sum[label] / class_sample_count[label]
            if label in updated_labels:
                Phi_nu[:, label] = args.m * Phi_nu[:, label] + (1 - args.m) * class_mean_feature
            else:
                Phi_nu[:, label] = class_mean_feature
                updated_labels.add(label)
        

def rdr_weights(feature, target, cls_num_list, Phi_nu, args):
    num_classes = len(cls_num_list)
    sample_weights = torch.zeros(target.size(0), dtype=torch.float, device=args.gpu)

    with torch.no_grad():
        for cls in range(num_classes):
            class_mask = (target == cls)      
            class_features = feature[class_mask]

            if class_features.size(0) > 0:
                Phi_de = class_features.T  
                Phi_deT_Phi_de = torch.matmul(Phi_de.T, Phi_de)
                try:
                    Phi_de_pinv = torch.inverse(Phi_deT_Phi_de)
                except RuntimeError as e:
                    Phi_de_pinv = torch.pinverse(Phi_deT_Phi_de)

                Phi_de_pinv_Phi_deT = torch.matmul(Phi_de_pinv, Phi_de.T)
                Phi_nu_cls = Phi_nu[:, cls].unsqueeze(1)

                r = torch.matmul(Phi_de_pinv_Phi_deT, Phi_nu_cls) * class_features.size(0)
                r = torch.clamp(r, min=1e-9) 

                sample_weights[class_mask] = r.squeeze()

            if args.log_results:
                wandb.log({f"class{cls}_r": r.squeeze().mean().item()})

        for cls in range(num_classes):
            class_mask = (target == cls)
            sample_weights[class_mask] *= (sum(cls_num_list) / cls_num_list[cls])

        # normalize
        sample_weights = sample_weights / sample_weights.sum() * sample_weights.size(0)
        
        if args.log_results:
            for cls in range(num_classes):
                class_mask = (target == cls)
                if sample_weights[class_mask].size(0) > 0: 
                    wandb.log({f"class{cls}_w": sample_weights[class_mask].mean().item()})

    return sample_weights


if __name__ == '__main__':
    main()
