import argparse
import os
import numpy as np
from tqdm import tqdm

from mypath import Path
from dataloaders import make_data_loader
from modeling.sync_batchnorm.replicate import patch_replication_callback
from modeling.deeplab import *
from utils.loss import SegmentationLosses
from utils.calculate_weights import calculate_weigths_labels
from utils.lr_scheduler import LR_Scheduler
from utils.saver import Saver
from utils.metrics import Evaluator

import distiller_reviewkd as distiller
import logging
import sys
import time
''' Setup a root logger by'''
logger = logging.getLogger(__package__)
logger.setLevel(logging.INFO)

class Trainer(object):
    def __init__(self, args):
        self.args = args

        # Define Saver
        self.saver = Saver(args)
        self.saver.save_experiment_config()

        ''' LOG'''
        self.logger = logger
        self.set_log_handler()
        # Define Dataloader
        kwargs = {'num_workers': args.workers, 'pin_memory': True}
        self.logger.info('===> Loading dataset...')
        self.train_loader, self.val_loader, self.test_loader, self.nclass = make_data_loader(args, **kwargs)

        # Define network
        self.t_net = DeepLab(num_classes=self.nclass,
                             backbone='resnet101',
                             output_stride=args.out_stride,
                             sync_bn=args.sync_bn,
                             freeze_bn=args.freeze_bn)
        checkpoint = torch.load(args.tnet)
        self.t_net.load_state_dict(checkpoint['state_dict'])

        self.s_net = DeepLab(num_classes=self.nclass,
                             backbone=args.backbone,
                             output_stride=args.out_stride,
                             sync_bn=args.sync_bn,
                             freeze_bn=args.freeze_bn)
        self.d_net = distiller.Distiller(self.t_net, self.s_net, self.args)

        # self.logger.info('Teacher Net: ')
        # self.logger.info(self.t_net)
        # self.logger.info('Student Net: ')
        # self.logger.info(self.s_net)
        # print('Teacher Net: ')
        # print(self.t_net)
        # print('Student Net: ')
        # print(self.s_net)
        self.logger.info('the number of teacher model parameters: {}'.format(
            sum([p.data.nelement() for p in self.t_net.parameters()])))
        self.logger.info('the number of student model parameters: {}'.format(
            sum([p.data.nelement() for p in self.s_net.parameters()])))

        self.distill_ratio = 1e-5
        self.batch_size = args.batch_size

        distill_params = [{'params': self.s_net.get_1x_lr_params(), 'lr': args.lr},
                          {'params': self.s_net.get_10x_lr_params(), 'lr': args.lr * 10},]
                        # {'params': self.d_net.get_embed_params(), 'lr': args.lr * 1}]

        # # Define Optimizer
        self.optimizer = torch.optim.SGD(distill_params, momentum=args.momentum,
                                         weight_decay=args.weight_decay, nesterov=args.nesterov)

        # Define Criterion
        # whether to use class balanced weights
        if args.use_balanced_weights:
            classes_weights_path = os.path.join(Path.db_root_dir(args.dataset), args.dataset+'_classes_weights.npy')
            if os.path.isfile(classes_weights_path):
                weight = np.load(classes_weights_path)
            else:
                weight = calculate_weigths_labels(args.dataset, self.train_loader, self.nclass)
            weight = torch.from_numpy(weight.astype(np.float32))
        else:
            weight = None
        self.criterion = SegmentationLosses(weight=weight, cuda=args.cuda).build_loss(mode=args.loss_type)
        
        # Define Evaluator
        self.evaluator = Evaluator(self.nclass)
        # Define lr scheduler
        self.scheduler = LR_Scheduler(args.lr_scheduler, args.lr,
                                    args.epochs, len(self.train_loader),
                                    lr_step=args.lr_step)

        # Using cuda
        if args.cuda:
            self.s_net = torch.nn.DataParallel(self.s_net).cuda()
            self.d_net = torch.nn.DataParallel(self.d_net).cuda()

        # Resuming checkpoint
        self.best_pred = 0.0
        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'" .format(args.resume))
            checkpoint = torch.load(args.resume)
            if args.cuda:
                self.s_net.module.load_state_dict(checkpoint['state_dict'])
            else:
                self.s_net.load_state_dict(checkpoint['state_dict'])
            self.logger.info("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        # Clear start epoch if fine-tuning
        if args.ft:
            args.start_epoch = 0

    def training(self, epoch):
        train_loss = 0.0
        train_seg_loss = 0.0
        train_distill_loss = 0.0
        self.d_net.train()
        self.d_net.module.t_net.train()
        self.d_net.module.s_net.train()
        tbar = tqdm(self.train_loader,bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
        num_img_tr = len(self.train_loader)

        optimizer = self.optimizer

        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']
            batch_size = image.shape[0]
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            self.scheduler(optimizer, i, epoch, self.best_pred)
            optimizer.zero_grad()
            output, loss_distill = self.d_net(image)

            loss_seg = self.criterion(output, target)
            '''Better capture seg loss and distillation loss'''
            loss_distill = loss_distill.sum() / batch_size * self.args.alpha
            loss = loss_seg + loss_distill

            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_seg_loss += loss_seg.item()
            train_distill_loss += loss_distill.item()
            tbar.set_description('Train Loss:{0:.2f},S:{1:.2f},D:{2:.2f}'.format((train_loss / (i + 1)),(train_seg_loss / (i + 1)),(train_distill_loss / (i + 1)))
                )

        # self.logger.info('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
        self.logger.info('Train Loss: {0:.3f}, Seg: {1:.3f}, Distill: {2:.3f}'.format(
        train_loss,train_seg_loss,train_distill_loss))

        if self.args.no_val:
            # save checkpoint every epoch
            is_best = False
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.s_net.module.state_dict(),
                'best_pred': self.best_pred,
            }, is_best)


    def validation(self, epoch):
        self.s_net.eval()
        self.evaluator.reset()
        tbar = tqdm(self.val_loader, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}', desc='\r')
        test_loss = 0.0
        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            with torch.no_grad():
                output = self.s_net(image)
            loss = self.criterion(output, target)
            test_loss += loss.item()
            tbar.set_description('Test loss: %.3f' % (test_loss / (i + 1)))
            pred = output.data.cpu().numpy()
            target = target.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            # Add batch sample into evaluator
            self.evaluator.add_batch(target, pred)

        # Fast test during the training
        Acc = self.evaluator.Pixel_Accuracy()
        Acc_class = self.evaluator.Pixel_Accuracy_Class()
        mIoU = self.evaluator.Mean_Intersection_over_Union()
        FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()
        # self.logger.info('********** Validation **********')
        # self.logger.info('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
        self.logger.info("Val: Loss:{0:.3f}, Acc:{1:.6f}, Acc_class:{2:.6f}, mIoU:{3:.6f}, fwIoU: {4:.6f}\n".format(test_loss, Acc, Acc_class, mIoU, FWIoU))
        # self.logger.info('Test Loss: %.3f' % test_loss)

        new_pred = mIoU
        if new_pred > self.best_pred:
            is_best = True
            self.best_pred = new_pred
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.s_net.module.state_dict(),
                
                'best_pred': self.best_pred,
            }, is_best)

    def set_log_handler(self):
        '''
            1. Set up formatter
            2. Set up filehandler to save log
            3. Set up streamhandler to show on the console
            4. All module logger will propagate their information to 
                this root logger.
        '''
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s',
                                    '%Y-%m-%d %H:%M:%S')
        file_handler=logging.FileHandler(os.path.join(self.saver.experiment_dir,'log.txt'))
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.addHandler(logging.StreamHandler(sys.stdout))

def main():
    parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus Training")
    parser.add_argument('--backbone', type=str, default='resnet18',
                        choices=['resnet101', 'resnet18', 'xception', 'drn', 'mobilenet'],
                        help='backbone name (default: resnet)')
    parser.add_argument('--out-stride', type=int, default=16,
                        help='network output stride (default: 8)')
    parser.add_argument('--dataset', type=str, default='pascal',
                        choices=['pascal', 'cocostuff10k'],
                        help='dataset name (default: pascal)')
    parser.add_argument('--use-sbd', action='store_true', default=False,
                        help='whether to use SBD dataset (default: False)')
    parser.add_argument('--workers', type=int, default=4,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--base-size', type=int, default=513,
                        help='base image size')
    parser.add_argument('--crop-size', type=int, default=513,
                        help='crop image size')
    parser.add_argument('--sync-bn', action='store_true', default=False,
                        help='whether to use sync bn (default: False)')
    parser.add_argument('--freeze-bn', action='store_true', default=False,
                        help='whether to freeze bn parameters (default: False)')
    parser.add_argument('--loss-type', type=str, default='ce',
                        choices=['ce', 'focal'],
                        help='loss func type (default: ce)')
    # training hyper params
    parser.add_argument('--epochs', type=int, default=None, metavar='N',
                        help='number of epochs to train (default: auto)')
    parser.add_argument('--start_epoch', type=int, default=0,
                        metavar='N', help='start epochs (default:0)')
    parser.add_argument('--batch-size', type=int, default=None,
                        metavar='N', help='input batch size for \
                                training (default: auto)')
    parser.add_argument('--test-batch-size', type=int, default=None,
                        metavar='N', help='input batch size for \
                                testing (default: auto)')
    parser.add_argument('--use-balanced-weights', action='store_true', default=False,
                        help='whether to use balanced weights (default: False)')
    # optimizer params
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (default: auto)')
    parser.add_argument('--lr-scheduler', type=str, default='poly',
                        choices=['poly', 'step', 'cos'],
                        help='lr scheduler mode: (default: poly)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        metavar='M', help='momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        metavar='M', help='w-decay (default: 5e-4)')
    parser.add_argument('--nesterov', action='store_true', default=False,
                        help='whether use nesterov (default: False)')
    # cuda, seed and logging
    parser.add_argument('--no-cuda', action='store_true', default=
                        False, help='disables CUDA training')
    parser.add_argument('--gpu-ids', type=str, default='0',
                        help='use which gpu to train, must be a \
                        comma-separated list of integers only (default=0)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    # checking point
    parser.add_argument('--resume', type=str, default=None,
                        help='put the path to resuming file if needed')
    parser.add_argument('--checkname', type=str, default=None,
                        help='set the checkpoint name')
    # finetuning pre-trained models
    parser.add_argument('--ft', action='store_true', default=False,
                        help='finetuning on a different dataset')
    # evaluation option
    parser.add_argument('--eval-interval', type=int, default=1,
                        help='evaluuation interval (default: 1)')
    parser.add_argument('--no-val', action='store_true', default=False,
                        help='skip validation during training')
    # attn
    parser.add_argument('--heads',type=int, default=None,
                        help='multi-head attention')
    parser.add_argument('--alpha',type=float, default=0.1,
                        help='distillation ratio (default=0.1)')
    parser.add_argument('--beta',type=float, default=0.5,
                        help='seq distillation ratio (default=0.5)')
    parser.add_argument('--gamma',type=float, default=0.5,
                        help='anchor distillation ratio (default=0.5)')
    parser.add_argument('-n',type=int, default=4,
                        help='n split patches')
    parser.add_argument('-m',type=int, default=4,
                        help='m split patches')
    parser.add_argument('--sp-h',type=int, default=0,
                        help='height of group patches')
    parser.add_argument('--sp-w',type=int, default=0,
                        help='width of group patches')
    parser.add_argument('--sp-hs',type=int, default=0,
                        help='h stride of group patches')
    parser.add_argument('--sp-ws',type=int, default=0,
                        help='w stride of group patches')
    parser.add_argument('--lr-step',type=int, default=20,
                        help='param for step scheduler')
    parser.add_argument('--anchor_h',type=int, default=129,
                        help='height for anchor distillation')
    parser.add_argument('--anchor_w',type=int, default=129,
                        help='width for anchor distillation')           
    # DEBUG option by
    parser.add_argument('--debug',action='store_true',default=False,
                        help='Fix log dir during debug per day')
    parser.add_argument('--flag', type=str, default=None,
                        help='Specify log flag')
    parser.add_argument('--attn-type', type=str, default=None,
                        choices=['stack_attn', 'batch_attn', 'anchor_attn','spatial_attn','None', None],
                        help='attn type')
    parser.add_argument('--tnet',type=str,default='pretrained/deeplab-resnet.pth.tar',
                        help='Pre-trained teacher model')
    # comp method
    parser.add_argument('--comp',type=str,default='kd',
                        choices=['kd','at','fitnet'],help='classical method')
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        try:
            # os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu_ids
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')

    # default settings for epochs, batch_size and lr
    if args.epochs is None:
        epoches = {
            'pascal': 50,
            'cocostuff10k':30,
        }
        args.epochs = epoches[args.dataset.lower()]

    if args.batch_size is None:
        args.batch_size = 6 * len(args.gpu_ids)

    if args.test_batch_size is None:
        args.test_batch_size = args.batch_size

    if args.lr is None:
        lrs = {
            'pascal': 0.007,
            'cocostuff10k':0.01,
        }
        args.lr = lrs[args.dataset.lower()] / (6 * len(args.gpu_ids)) * args.batch_size


    if args.checkname is None:
        args.checkname = 'deeplab-'+str(args.backbone)
    # print(args)
    torch.manual_seed(args.seed)
    trainer = Trainer(args)
    start_time = time.time()
    trainer.logger.info('Running file: {}'.format(__file__))
    trainer.logger.info('***** MODEL PARAMETER *****')
    trainer.logger.info(args)
    trainer.logger.info('Starting Epoch: {}'.format(trainer.args.start_epoch))
    trainer.logger.info('Total Epoches: {}'.format(trainer.args.epochs))
    trainer.logger.info('===> Start training. Log dir is {}.'.format(trainer.saver.experiment_dir))
    for epoch in range(trainer.args.start_epoch, trainer.args.epochs):
        trainer.training(epoch)
        if not trainer.args.no_val and epoch % args.eval_interval == (args.eval_interval - 1):
            trainer.validation(epoch)
    end_time = time.time()
    trainer.logger.info('===> Finished! Log dir is {}. Total time cost {}'.format(trainer.saver.experiment_dir, time.strftime("%H:%M:%S",time.gmtime(end_time-start_time))))

if __name__ == "__main__":
    main()
