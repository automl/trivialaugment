import itertools
import json
import logging
import math
import os
from collections import OrderedDict
import gc
import tempfile
import pickle
from dataclasses import dataclass
import random
from time import time

import numpy as np
import torch
from torch import nn, optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import transforms

from tqdm import tqdm
import yaml
from theconf import Config as C, ConfigArgumentParser
from argparse import ArgumentParser

from TrivialAugment.common import get_logger
from TrivialAugment.data import get_dataloaders
from TrivialAugment.lr_scheduler import adjust_learning_rate_resnet
from TrivialAugment.metrics import accuracy, Accumulator
from TrivialAugment.networks import get_model, num_class
from warmup_scheduler import GradualWarmupScheduler
import aug_lib

logger = get_logger('TrivialAugment')
logger.setLevel(logging.DEBUG)

def run_epoch(rank, worldsize, model, loader, loss_fn, optimizer, desc_default='', epoch=0, writer=None, verbose=1, scheduler=None,sample_pairing_loader=None):
    tqdm_disable = bool(os.environ.get('TASK_NAME', ''))    # KakaoBrain Environment
    if verbose:
        logging_loader = tqdm(loader, disable=tqdm_disable)
        logging_loader.set_description('[%s %04d/%04d]' % (desc_default, epoch, C.get()['epoch']))
    else:
        logging_loader = loader

    metrics = Accumulator()
    cnt = 0
    eval_cnt = 0
    total_steps = len(loader)
    steps = 0

    gc.collect()
    torch.cuda.empty_cache()
    #print('mem usage', resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    communicate_grad_every = C.get().get('communicate_grad_every', 1)
    before_load_time = time()
    if C.get().get('load_sample_pairing_batch',False) and sample_pairing_loader is not None:
        sample_pairing_iter = iter(sample_pairing_loader)
        aug_lib.blend_images = [transforms.ToPILImage()(sample_pairing_loader.denorm(ti)) for ti in
                                next(sample_pairing_iter)[0]]
    for batch in logging_loader: # logging loader might be a loader or a loader wrapped into tqdm
        data, label = batch[:2]
        steps += 1
        if C.get().get('load_sample_pairing_batch',False) and sample_pairing_loader is not None:
            try:
                aug_lib.blend_images = [transforms.ToPILImage()(sample_pairing_loader.denorm(ti)) for ti in next(sample_pairing_iter)[0]]
            except StopIteration:
                print("Blend images iterator ended. If this is printed twice per loop, there is something out-of-order.")
                pass
        if worldsize > 1:
            data, label = data.to(rank), label.to(rank)
        else:
            data, label = data.cuda(), label.cuda()


        communicate_grad = steps % communicate_grad_every == 0
        just_communicated_grad = steps % communicate_grad_every == 1 # also is true in first step of each epoch
        if optimizer and (communicate_grad_every == 1 or just_communicated_grad):
            optimizer.zero_grad()

        preds = model(data)
        loss = loss_fn(preds, label)
        if optimizer:
            if communicate_grad:
                loss.backward()
            else:
                with model.no_sync():
                    loss.backward()

            if C.get()['optimizer'].get('clip', 5) > 0:
                nn.utils.clip_grad_norm_(model.parameters(), C.get()['optimizer'].get('clip', 5))
            if (steps-1) % C.get().get('step_optimizer_every', 1) == C.get().get('step_optimizer_nth_step', 0): # default is to step on the first step of each pack
                optimizer.step()
        #print(f"Time for forward/backward {time()-fb_time}")
        top1, top5 = accuracy(preds, label, (1, 5))
        metrics.add_dict({
            'loss': loss.item() * len(data),
            'top1': top1.item() * len(data),
            'top5': top5.item() * len(data),
        })
        if steps % 2 == 0:
            metrics.add('eval_top1', top1.item() * len(data)) # times 2 since it is only recorded every sec step
            eval_cnt += len(data)
        cnt += len(data)
        if verbose:
            postfix = metrics.divide(cnt, eval_top1=eval_cnt)
            if optimizer:
                postfix['lr'] = optimizer.param_groups[0]['lr']
            logging_loader.set_postfix(postfix)

        if scheduler is not None:
            scheduler.step(epoch - 1 + float(steps) / total_steps)

        #before_load_time = time()
        del preds, loss, top1, top5, data, label

    if tqdm_disable:
        if optimizer:
            logger.info('[%s %03d/%03d] %s lr=%.6f', desc_default, epoch, C.get()['epoch'],  metrics.divide(cnt, eval_top1=eval_cnt), optimizer.param_groups[0]['lr'])
        else:
            logger.info('[%s %03d/%03d] %s', desc_default, epoch, C.get()['epoch'], metrics.divide(cnt, eval_top1=eval_cnt))

    metrics = metrics.divide(cnt, eval_top1=eval_cnt)
    if optimizer:
        metrics.metrics['lr'] = optimizer.param_groups[0]['lr']
    if verbose:
        for key, value in metrics.items():
            writer.add_scalar(key, value, epoch)
    return metrics


def train_and_eval(rank, worldsize, tag, dataroot, test_ratio=0.0, cv_fold=0, reporter=None, metric='last', save_path=None, only_eval=False):
    if not reporter:
        reporter = lambda **kwargs: 0

    if not tag or (worldsize and torch.distributed.get_rank() > 0):
        from TrivialAugment.metrics import SummaryWriterDummy as SummaryWriter
        logger.warning('tag not provided or rank > 0 -> no tensorboard log.')
    else:
        from tensorboardX import SummaryWriter

    os.makedirs('./logs/', exist_ok=True)
    writers = [SummaryWriter(log_dir='./logs/%s/%s' % (tag, x)) for x in ['train', 'valid', 'test', 'testtrain']]

    aug_lib.set_augmentation_space(C.get().get('augmentation_search_space', 'standard'), C.get().get('augmentation_parameter_max', 30), C.get().get('custom_search_space_augs', None))
    max_epoch = C.get()['epoch']
    trainsampler, trainloader, validloader, testloader_, testtrainloader_, dataset_info = get_dataloaders(C.get()['dataset'], C.get()['batch'], dataroot, test_ratio, split_idx=cv_fold, distributed=worldsize>1, started_with_spawn=C.get()['started_with_spawn'], summary_writer=writers[0])

    # create a model & an optimizer
    model = get_model(C.get()['model'], C.get()['batch'], num_class(C.get()['dataset']), writer=writers[0])
    if worldsize > 1:
        model = DDP(model.to(rank), device_ids=[rank])
    else:
        model = model.to('cuda:0')


    criterion = nn.CrossEntropyLoss()
    if C.get()['optimizer']['type'] == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=C.get()['lr'],
            momentum=C.get()['optimizer'].get('momentum', 0.9),
            weight_decay=C.get()['optimizer']['decay'],
            nesterov=C.get()['optimizer']['nesterov']
        )
    elif C.get()['optimizer']['type'] == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=C.get()['lr'],
            betas=(C.get()['optimizer'].get('momentum',.9),.999)
        )
    else:
        raise ValueError('invalid optimizer type=%s' % C.get()['optimizer']['type'])


    lr_scheduler_type = C.get()['lr_schedule'].get('type', 'cosine')
    if lr_scheduler_type == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=C.get()['epoch'], eta_min=0.)
    elif lr_scheduler_type == 'resnet':
        scheduler = adjust_learning_rate_resnet(optimizer)
    elif lr_scheduler_type == 'constant':
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda e: 1.)
    else:
        raise ValueError('invalid lr_schduler=%s' % lr_scheduler_type)

    if C.get()['lr_schedule'].get('warmup', None):
        scheduler = GradualWarmupScheduler(
            optimizer,
            multiplier=C.get()['lr_schedule']['warmup']['multiplier'],
            total_epoch=C.get()['lr_schedule']['warmup']['epoch'],
            after_scheduler=scheduler
        )

    result = OrderedDict()
    epoch_start = 1
    if save_path and os.path.exists(save_path):
        logger.info('%s file found. loading...' % save_path)
        data = torch.load(save_path, map_location='cpu')
        if 'model' in data or 'state_dict' in data:
            key = 'model' if 'model' in data else 'state_dict'
            logger.info('checkpoint epoch@%d' % data['epoch'])
            if C.get().get('load_main_model', False):
                model.load_state_dict(data[key])
                #if not isinstance(model, DataParallel):
                    #model.load_state_dict({k.replace('module.', ''): v for k, v in data[key].items()})
                #else:
                    #model.load_state_dict({k if 'module.' in k else 'module.'+k: v for k, v in data[key].items()})
                optimizer.load_state_dict(data['optimizer'])
                if data['epoch'] < C.get()['epoch']:
                    epoch_start = data['epoch'] + 1
                else:
                    only_eval = True
        else:
            #model.load_state_dict({k: v for k, v in data.items()})
            raise ValueError(f"Wrong format of data in save path: {save_path}.")
        del data
    else:
        logger.info('"%s" file not found. skip to pretrain weights...' % save_path)
        if only_eval:
            logger.warning('model checkpoint not found. only-evaluation mode is off.')
        only_eval = False

    if only_eval:
        logger.info('evaluation only+')
        model.eval()
        rs = dict()
        with torch.no_grad():
            rs['train'] = run_epoch(rank, worldsize, model, trainloader, criterion, None, desc_default='train', epoch=0, writer=writers[0])
            #rs['valid'] = run_epoch(rank, worldsize, model, validloader, criterion, None, desc_default='valid', epoch=0, writer=writers[1])
            rs['test'] = run_epoch(rank, worldsize, model, testloader_, criterion, None, desc_default='*test', epoch=0, writer=writers[2])
        for key, setname in itertools.product(['loss', 'top1', 'top5'], ['train', 'test']):
            if setname not in rs:
                continue
            result['%s_%s' % (key, setname)] = rs[setname][key]
        result['epoch'] = 0
        return result

    # train loop
    best_top1 = 0
    for epoch in range(epoch_start, max_epoch + 1):
        if worldsize > 1:
            trainsampler.set_epoch(epoch)

        model.train()
        rs = dict()
        rs['train'] = run_epoch(rank, worldsize,model, trainloader, criterion, optimizer, desc_default='train', epoch=epoch, writer=writers[0], verbose=True, scheduler=scheduler, sample_pairing_loader=testtrainloader_)
        model.eval()

        if math.isnan(rs['train']['loss']):
            raise Exception('train loss is NaN.')

        if epoch % 20 == 0 or epoch == max_epoch:
            with torch.no_grad():
                if C.get().get('compute_testtrain', False):
                    rs['testtrain'] = run_epoch(rank, worldsize, model, testtrainloader_, criterion, None, desc_default='testtrain', epoch=epoch, writer=writers[3], verbose=True)
                rs['test'] = run_epoch(rank, worldsize, model, testloader_, criterion, None, desc_default='*test', epoch=epoch, writer=writers[2], verbose=True)


            if metric == 'last' or rs[metric]['top1'] > best_top1:
                if metric != 'last':
                    best_top1 = rs[metric]['top1']
                for key, setname in itertools.product(['loss', 'top1', 'top5'], ['train', 'test', 'testtrain']):
                    if setname in rs and key in rs[setname]:
                        result['%s_%s' % (key, setname)] = rs[setname][key]
                result['epoch'] = epoch

                #writers[1].add_scalar('valid_top1/best', rs['valid']['top1'], epoch)
                writers[2].add_scalar('test_top1/best', rs['test']['top1'], epoch)

                reporter(
                    loss_valid=rs['test']['loss'], top1_valid=rs['test']['top1'],
                    loss_test=rs['test']['loss'], top1_test=rs['test']['top1']
                )

                # save checkpoint
                if save_path and C.get().get('save_model', True) and (worldsize <= 1 or torch.distributed.get_rank() == 0):
                    logger.info('save model@%d to %s' % (epoch, save_path))
                    torch.save({
                        'epoch': epoch,
                        'log': {
                            'train': rs['train'].get_dict(),
                            'test': rs['test'].get_dict(),
                        },
                        'optimizer': optimizer.state_dict(),
                        'model': model.state_dict()
                    }, save_path)
                    torch.save({
                        'epoch': epoch,
                        'log': {
                            'train': rs['train'].get_dict(),
                            'test': rs['test'].get_dict(),
                        },
                        'optimizer': optimizer.state_dict(),
                        'model': model.state_dict()
                    }, save_path.replace('.pth', '_e%d_top1_%.3f_%.3f' % (epoch, rs['train']['top1'], rs['test']['top1']) + '.pth'))

        early_finish_epoch = C.get().get('early_finish_epoch', None)
        if early_finish_epoch == epoch:
            break

    del model

    return result

def setup(global_rank, local_rank, world_size, port_suffix):
    torch.cuda.set_device(local_rank)
    if port_suffix is not None:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = f'12{port_suffix}'

        # initialize the process group
        dist.init_process_group("nccl", rank=global_rank, world_size=world_size)
        return global_rank, world_size
    else:
        dist.init_process_group(backend='NCCL', init_method='env://')
        return torch.distributed.get_rank(), torch.distributed.get_world_size()

def cleanup():
    dist.destroy_process_group()

def parse_args():
    parser = ConfigArgumentParser(conflict_handler='resolve')
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--dataroot', type=str, default='/data/private/pretrainedmodels',
                        help='torchvision data folder')
    parser.add_argument('--save', type=str, default='')
    parser.add_argument('--cv-ratio', type=float, default=0.0)
    parser.add_argument('--cv', type=int, default=0)
    parser.add_argument('--only-eval', action='store_true')
    parser.add_argument('--local_rank', default=None, type=int)
    return parser.parse_args()


def spawn_process(global_rank, worldsize, port_suffix, args, config_path=None, communicate_results_with_queue=None, local_rank=None,):
    if local_rank is None:
        local_rank = global_rank
    started_with_spawn = worldsize is not None and worldsize > 0
    if worldsize != 0:
        global_rank, worldsize = setup(global_rank, local_rank, worldsize, port_suffix)
    print('dist info', local_rank,global_rank,worldsize)
    #communicate_results_with_queue.value = 1.
    #return
    if config_path is not None:
        C(config_path)
    C.get()['started_with_spawn'] = started_with_spawn

    if worldsize:
        assert worldsize == C.get()['gpus'], f"Did not specify the number of GPUs in Config with which it was started: {worldsize} vs {C.get()['gpus']}"
    else:
        assert 'gpus' not in C.get() or C.get()['gpus'] == 1

    assert (args.only_eval and args.save) or not args.only_eval, 'checkpoint path not provided in evaluation mode.'

    if not args.only_eval:
        if args.save:
            logger.info('checkpoint will be saved at %s' % args.save)
        else:
            logger.warning('Provide --save argument to save the checkpoint. Without it, training result will not be saved!')

    #if args.save:
        #add_filehandler(logger, args.save.replace('.pth', '.log'))

    #logger.info(json.dumps(C.get().conf, indent=4))
    if 'seed' in C.get():
        seed = C.get()['seed']
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        #torch.backends.cudnn.benchmark = False


    import time
    t = time.time()
    result = train_and_eval(local_rank, worldsize, args.tag, args.dataroot, test_ratio=args.cv_ratio, cv_fold=args.cv, save_path=args.save, only_eval=args.only_eval, metric='last')
    elapsed = time.time() - t
    print('done')

    logger.info(f'done on rank {global_rank}.')
    logger.info('model: %s' % C.get()['model'])
    logger.info('augmentation: %s' % C.get()['aug'])
    logger.info('\n' + json.dumps(result, indent=4))
    logger.info('elapsed time: %.3f Hours' % (elapsed / 3600.))
    logger.info('top1 error in testset: %.4f' % (1. - result['top1_test']))
    logger.info(args.save)
    if worldsize:
        cleanup()

    if global_rank == 0 and communicate_results_with_queue is not None:
        #communicate_results_with_queue.put([result])
        communicate_results_with_queue.value = result['top1_test']


@dataclass
class Args:
    tag: str = ''
    dataroot: str = None
    save: str = ''
    cv_ratio: float = 0.
    cv: int = 0
    only_eval: bool = False
    local_rank: None = None

def run_from_py(dataroot, config_dict, save=''):
    args = Args(dataroot=dataroot, save=save)
    with tempfile.NamedTemporaryFile(mode='w+') as f, tempfile.NamedTemporaryFile() as result_file:
        path = f.name
        yaml.dump(config_dict, f)
        world_size = torch.cuda.device_count()
        port_suffix = str(random.randint(100, 999))
        #result_queue = mp.get_context('spawn').Queue()
        result_queue = mp.get_context('spawn').Value('d',.0)
        if world_size > 1:
            outcome = mp.spawn(spawn_process,
                               args=(world_size, port_suffix, args, path, result_queue),
                               nprocs=world_size,
                               join=True)
        else:
            outcome = spawn_process(0, 0, port_suffix, args, path, result_queue)
        #result = result_queue.get()[0]
        result = result_queue.value
    return result


if __name__ == '__main__':
    pre_parser = ArgumentParser()
    pre_parser.add_argument('--local_rank', default=None, type=int)
    args, _ = pre_parser.parse_known_args()
    if args.local_rank is None:
        print("Spawning processes")
        world_size = torch.cuda.device_count()
        port_suffix = str(random.randint(10,99))
        if world_size > 1:
            outcome = mp.spawn(spawn_process,
                              args=(world_size,port_suffix,parse_args()),
                              nprocs=world_size,
                              join=True)
        else:
            spawn_process(0, 0, None, parse_args())
        with open(f'/tmp/samshpopt/training_with_portsuffix_{port_suffix}.pkl', 'r') as f:
            result = pickle.load(f)
    else:
        spawn_process(None, -1, None, parse_args(), local_rank=args.local_rank)
