import os
import shutil
import torch
from collections import OrderedDict
import glob
import time
class Saver(object):

    def __init__(self, args):
        self.args = args
        self.timesample = time.strftime('%Y-%m-%d')
        self.directory = os.path.join('run', args.dataset, args.checkname)
        if args.debug:
            # if debug, just use the same log dir per day...
            self.experiment_dir = os.path.join(self.directory,self.timesample+'-debug')
        elif args.flag:
            #if flag, use timesample+flag for log name...
            self.experiment_dir = os.path.join(self.directory, self.timesample+'-'+args.flag)
        else:
            self.runs = sorted(glob.glob(os.path.join(self.directory, self.timesample+'_*')))
            run_id = int(self.runs[-1].split('_')[-1]) + 1 if self.runs else 0
            self.experiment_dir = os.path.join(self.directory, self.timesample+'_{}'.format(str(run_id)))
            
        if not os.path.exists(self.experiment_dir):
            os.makedirs(self.experiment_dir)
        self.regular_ckp_dir=os.path.join(self.experiment_dir,'regular/')
        if not os.path.exists(self.regular_ckp_dir):
            os.makedirs(self.regular_ckp_dir)

    def save_checkpoint(self, state, is_best, filename='checkpoint.pth.tar'):
        """Saves checkpoint to disk"""
#        filename = os.path.join(self.experiment_dir, filename)
#        torch.save(state, filename)
#        if is_best:
#            best_pred = state['best_pred']
#            with open(os.path.join(self.experiment_dir, 'best_pred.txt'), 'w') as f:
#                f.write(str(best_pred))
#            if self.runs:
#                previous_miou = [0.0]
#                for run in self.runs:
#                    run_id = run.split('_')[-1]
#                    path = os.path.join(self.directory, 'experiment_{}'.format(str(run_id)), 'best_pred.txt')
#                    if os.path.exists(path):
#                        with open(path, 'r') as f:
#                            miou = float(f.readline())
#                            previous_miou.append(miou)
#                    else:
#                        continue
#                max_miou = max(previous_miou)
#                if best_pred > max_miou:
#                    shutil.copyfile(filename, os.path.join(self.directory, 'model_best.pth.tar'))
#            else:
#                shutil.copyfile(filename, os.path.join(self.directory, 'model_best.pth.tar'))
        if is_best:
            best_pred = state['best_pred']
            with open(os.path.join(self.experiment_dir, 'best_pred.txt'), 'w') as f:
                f.write(str(best_pred))
            pre_best_pth = sorted(glob.glob(os.path.join(self.experiment_dir,'*.pth.tar')))
            for f in pre_best_pth:
                os.remove(f)
            filename = os.path.join(self.experiment_dir,'checkpoint_%.4f.pth.tar'%float(best_pred))
        else:
            filename = os.path.join(self.experiment_dir, filename)
        torch.save(state, filename)
    def save_checkpoint_regular(self, state, filename='checkpoint.pth.tar'):
        epoch = state['epoch']
        regular_name = os.path.join(self.regular_ckp_dir,'checkpoint_%d.pth.tar'%int(epoch))
        link_name = os.path.join(self.regular_ckp_dir,'checkpoint.pth.tar')
        #if os.path.exists(os.path.join(self.regular_ckp_dir,'checkpoint.pth.tar')): #delete symbolic link
            #os.unlink(os.path.join(self.regular_ckp_dir,'checkpoint.pth.tar'))
        if os.path.exists(link_name): #delete symbolic link
            os.remove(link_name)
        for f in os.listdir(self.regular_ckp_dir):
            #if os.path.isfile(os.path.join(self.regular_ckp_dir,f)):
            os.remove(os.path.join(self.regular_ckp_dir,f))
        torch.save(state,regular_name)
        os.symlink(regular_name,link_name)

    def save_experiment_config(self):
        logfile = os.path.join(self.experiment_dir, 'parameters.txt')
        log_file = open(logfile, 'w')
        p = OrderedDict()
        p['datset'] = self.args.dataset
        p['backbone'] = self.args.backbone
        p['out_stride'] = self.args.out_stride
        p['lr'] = self.args.lr
        p['lr_scheduler'] = self.args.lr_scheduler
        p['loss_type'] = self.args.loss_type
        p['epoch'] = self.args.epochs
        p['base_size'] = self.args.base_size
        p['crop_size'] = self.args.crop_size

        for key, val in p.items():
            log_file.write(key + ':' + str(val) + '\n')
        log_file.close()
