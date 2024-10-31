from copy import deepcopy
import os
import sys
import json
import warnings
warnings.filterwarnings("ignore")

import math
import time
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torchvision import transforms
from util.icbhi_dataset import ICBHIDataset
from util.icbhi_util import get_score
from util.augmentation import SpecAugment
from util.misc import adjust_learning_rate, warmup_learning_rate, set_optimizer, update_moving_average
from util.misc import AverageMeter, accuracy, save_model, update_json
from models import get_backbone_class, Projector 
from models.caast import DualAttentionAST
from method import PatchMixLoss, PatchMixConLoss
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd
from collections import defaultdict
import torch.nn.functional as F

torch.cuda.set_device(0) 
torch.autograd.set_detect_anomaly(True)



        
def debug_info(name, tensor):
    """Helper function to print debugging information"""
    if tensor.requires_grad:
        if tensor.grad is not None:
            print(f"{name} - grad stats: min={tensor.grad.min().item():.4f}, max={tensor.grad.max().item():.4f}, mean={tensor.grad.mean().item():.4f}, has_nan={torch.isnan(tensor.grad).any().item()}")
    print(f"{name} - value stats: min={tensor.min().item():.4f}, max={tensor.max().item():.4f}, mean={tensor.mean().item():.4f}, has_nan={torch.isnan(tensor).any().item()}")

   

            
def parse_args():
    parser = argparse.ArgumentParser('argument for supervised training')

    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--print_freq', type=int, default=100)
    parser.add_argument('--save_freq', type=int, default=100)
    parser.add_argument('--save_dir', type=str, default='./save')
    parser.add_argument('--tag', type=str, default='',
                        help='tag for experiment name')
    parser.add_argument('--resume', type=str, default=None,
                        help='path of model checkpoint to resume')
    parser.add_argument('--eval', action='store_true',
                        help='only evaluation with pretrained encoder and classifier')
    parser.add_argument('--two_cls_eval', action='store_true',
                        help='evaluate with two classes')
    
    # optimization
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--lr_decay_epochs', type=str, default='120,160')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument('--warm_epochs', type=int, default=0,
                        help='warmup epochs')
    # dataset
    parser.add_argument('--dataset', type=str, default='icbhi')
    parser.add_argument('--data_folder', type=str, default='./data/icbhi_dataset')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=8)
    # icbhi dataset
    parser.add_argument('--class_split', type=str, default='lungsound',
                        help='lungsound: (normal, crackles, wheezes, both), diagnosis: (healthy, chronic diseases, non-chronic diseases)')
    parser.add_argument('--n_cls', type=int, default=0,
                        help='set k-way classification problem for class')
    parser.add_argument('--d_cls', type=int, default=0,
                        help='set k-way classification problem for device (meta)')
    parser.add_argument('--test_fold', type=str, default='official', choices=['official', '0', '1', '2', '3', '4'],
                        help='test fold to use official 60-40 split or 80-20 split from RespireNet')
    parser.add_argument('--sample_rate', type=int,  default=16000, 
                        help='sampling rate when load audio data, and it denotes the number of samples per one second')
    parser.add_argument('--desired_length', type=int,  default=8, 
                        help='fixed length size of individual cycle')
    parser.add_argument('--n_mels', type=int, default=128,
                        help='the number of mel filter banks')
    parser.add_argument('--pad_types', type=str,  default='repeat', 
                        help='zero: zero-padding, repeat: padding with duplicated samples, aug: padding with augmented samples')
    parser.add_argument('--resz', type=float, default=1, 
                        help='resize the scale of mel-spectrogram')
    parser.add_argument('--raw_augment', type=int, default=0, 
                        help='control how many number of augmented raw audio samples')
    parser.add_argument('--specaug_policy', type=str, default='icbhi_ast_sup', 
                        help='policy (argument values) for SpecAugment')
    parser.add_argument('--specaug_mask', type=str, default='mean', 
                        help='specaug mask value', choices=['mean', 'zero'])
    parser.add_argument('--nospec', action='store_true')

    # model
    parser.add_argument('--model', type=str, default='ast')
    parser.add_argument('--pretrained', action='store_true')
    
    parser.add_argument('--pretrained_ckpt', type=str, default=None,
                        help='path to pre-trained encoder model')
    parser.add_argument('--from_sl_official', action='store_true',
                        help='load from supervised imagenet-pretrained model (official PyTorch)')
    parser.add_argument('--ma_update', action='store_true',
                        help='whether to use moving average update for model')
    parser.add_argument('--ma_beta', type=float, default=0,
                        help='moving average value')
    # for AST
    parser.add_argument('--audioset_pretrained', action='store_true',
                        help='load from imagenet- and audioset-pretrained model')
    # for SSAST
    parser.add_argument('--ssast_task', type=str, default='ft_avgtok', 
                        help='pretraining or fine-tuning task', choices=['ft_avgtok', 'ft_cls'])
    parser.add_argument('--fshape', type=int, default=16, 
                        help='fshape of SSAST')
    parser.add_argument('--tshape', type=int, default=16, 
                        help='tshape of SSAST')
    parser.add_argument('--ssast_pretrained_type', type=str, default='Patch', 
                        help='pretrained ckpt version of SSAST model')

    parser.add_argument('--method', type=str, default='ce')

    # Meta Domain CL & Patch-Mix CL loss
    parser.add_argument('--proj_dim', type=int, default=768)
    parser.add_argument('--mix_beta', default=1.0, type=float,
                        help='patch-mix interpolation coefficient')
    parser.add_argument('--time_domain', action='store_true',
                        help='patchmix for the specific time domain')

    parser.add_argument('--temperature', type=float, default=0.06)
    parser.add_argument('--alpha', type=float, default=0.2)
    parser.add_argument('--ce_weight', type=float, default=1)
    parser.add_argument('--weighted_loss', action='store_true',
                        help='weighted cross entropy loss (higher weights on abnormal class)')
    parser.add_argument('--negative_pair', type=str, default='all',
                        help='the method for selecting negative pair', choices=['all', 'diff_label'])
    parser.add_argument('--target_type', type=str, default='grad_block',
                        help='how to make target representation', choices=['project_flow', 'grad_block1', 'grad_flow1', 'project_block1', 'grad_block2', 'grad_flow2', 'project_block2', 'project_block_all', 'representation_all', 'grad_block', 'grad_flow', 'project_block'])
    
    # Meta for SCL
    parser.add_argument('--device_mode', type=str, default='none',
                        help='the meta information for selecting', choices=['none', 'mixed'])
    
    # TSNE
    parser.add_argument('--visualize_embeddings', action='store_true',
                    help='visualize initial embeddings by domain and class before training')
    
    # ROC
    parser.add_argument('--roc', action='store_true')
    parser.add_argument('--fold_number', type=int, default='0')
    

    #confusion matrix
    parser.add_argument('--confusion_matrix', action='store_true',
                    help='plot and save confusion matrix during evaluation')
    
    #INTERSPEECH
    parser.add_argument('--fusion_type', type=str, default='concat',
                        help='fusion type', choices=['concat', 'gate'])
    
                        
    args = parser.parse_args()

    iterations = args.lr_decay_epochs.split(',')
    args.lr_decay_epochs = list([])
    for it in iterations:
        args.lr_decay_epochs.append(int(it))
    args.model_name = '{}_{}_{}'.format(args.dataset, args.model, args.method)
    if args.tag:
        args.model_name += '_{}'.format(args.tag)
    
    args.save_folder = os.path.join(args.save_dir, args.model_name)
    
    if not os.path.isdir(args.save_folder):
        os.makedirs(args.save_folder)

    if args.warm:
        args.warmup_from = args.learning_rate * 0.1
        args.warm_epochs = 10
        if args.cosine:
            eta_min = args.learning_rate * (args.lr_decay_rate ** 3)
            args.warmup_to = eta_min + (args.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * args.warm_epochs / args.epochs)) / 2
        else:
            args.warmup_to = args.learning_rate

            
    if args.dataset == 'icbhi':
        if args.class_split == 'lungsound':
                    
            if args.n_cls == 4:
                args.cls_list = ['normal', 'crackle', 'wheeze', 'both']
            elif args.n_cls == 2:
                args.cls_list = ['normal', 'abnormal']
            
            args.device_list = ['L', 'A', 'M', '3']
                
        elif args.class_split == 'diagnosis':
            if args.n_cls == 3:
                args.cls_list = ['healthy', 'chronic_diseases', 'non-chronic_diseases']
            elif args.n_cls == 2:
                args.cls_list = ['healthy', 'unhealthy']
            else:
                raise NotImplementedError
        
    else:
        raise NotImplementedError
    
    if args.n_cls == 0 and args.m_cls !=0:
        args.n_cls = args.m_cls
        args.cls_list = args.meta_cls_list

    return args

def set_loader(args):
    if args.dataset == 'icbhi':        
        args.h = int(args.desired_length * 100 - 2)
        # args.h = int(608)
        args.w = 128
        #args.h, args.w = 798, 128
        train_transform = [transforms.ToTensor(),
                            transforms.Resize(size=(int(args.h * args.resz), int(args.w * args.resz)))]
        val_transform = [transforms.ToTensor(),
                        transforms.Resize(size=(int(args.h * args.resz), int(args.w * args.resz)))]
        ##
        train_transform = transforms.Compose(train_transform)
        val_transform = transforms.Compose(val_transform)
        

        train_dataset = ICBHIDataset(train_flag=True, transform=train_transform, args=args, print_flag=True)
        val_dataset = ICBHIDataset(train_flag=False, transform=val_transform, args=args,  print_flag=True)
        
        args.class_nums = train_dataset.class_nums
        

        
    else:
        raise NotImplemented    
    
    


    # DataLoader 설정
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,shuffle=True,
                                                    num_workers=args.num_workers,pin_memory=False,
                                                    persistent_workers=True,
                                                    prefetch_factor=2, drop_last=True)
    
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                             num_workers=args.num_workers, pin_memory=True)

    
    
    return train_loader, val_loader, args


        
        
def set_model(args):
    kwargs = {}
    if args.model == 'ast':
        kwargs['input_fdim'] = int(args.h * args.resz)
        kwargs['input_tdim'] = int(args.w * args.resz)
        kwargs['label_dim'] = args.n_cls
        kwargs['imagenet_pretrain'] = args.from_sl_official
        kwargs['audioset_pretrain'] = args.audioset_pretrained
        kwargs['mix_beta'] = args.mix_beta  # for Patch-MixCL
        model = get_backbone_class(args.model)(**kwargs)
       
    elif args.model == 'proto':  # New model type
        print("Generating Dual Attention AST model")
        model = DualAttentionAST(
            label_dim=args.n_cls,
            fstride=10,
            tstride=10,
            input_fdim=int(args.h * args.resz),
            input_tdim=int(args.w * args.resz),
            imagenet_pretrain=args.from_sl_official,
            audioset_pretrain=args.audioset_pretrained,
            fusion_type=args.fusion_type  # 'concat' or 'gate'
        )
        
    else:   
        model = get_backbone_class(args.model)(**kwargs)
        
   # Classifier 설정
    if args.model == 'proto':
        # For dual_ast with concatenated features
        input_dim = (model.original_embedding_dim * 3 
                    if args.fusion_type == 'concat' 
                    else model.original_embedding_dim)
        
        classifier = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, args.n_cls)
        ).cuda()
        
       
    else:
        classifier = (nn.Linear(model.final_feat_dim, args.n_cls) 
                     if args.model not in ['ast', 'ssast','caast'] 
                     else deepcopy(model.mlp_head)).cuda()
        
    projector = Projector(model.final_feat_dim, args.proj_dim) if args.method in ['patchmix_cl'] else nn.Identity()
        
    
    if not args.weighted_loss:
        weights = None
        criterion = nn.CrossEntropyLoss()
    else: 
        weights = torch.tensor(args.class_nums, dtype=torch.float32)
        weights = 1.0 / (weights / weights.sum())
        weights /= weights.sum()
    
        criterion = nn.CrossEntropyLoss(weight=weights)

        
           
    if args.model not in ['ast', 'ssast','proto'] and args.from_sl_official:
        model.load_sl_official_weights()
        print('pretrained model loaded from PyTorch ImageNet-pretrained')
        
    

    # load SSL pretrained checkpoint for linear evaluation
    if args.pretrained and args.pretrained_ckpt is not None:
        ckpt = torch.load(args.pretrained_ckpt, map_location='cpu')
        state_dict = ckpt['model']

        # HOTFIX: always use dataparallel during SSL pretraining
        new_state_dict = {}
        for k, v in state_dict.items():
            if "module." in k:
                k = k.replace("module.", "")
            if "backbone." in k:
                k = k.replace("backbone.", "")
            if not 'mlp_head' in k: #del mlp_head
                new_state_dict[k] = v
        state_dict = new_state_dict
        model.load_state_dict(state_dict, strict=False)
        
        
        if ckpt.get('classifier', None) is not None:
            print("correct")

            classifier.load_state_dict(ckpt['classifier'], strict=True)

        print('pretrained model loaded from: {}'.format(args.pretrained_ckpt))
        
        
    if args.method == 'ce' or args.method == 'proto':
        criterion = [criterion.cuda()]
    elif args.method == 'patchmix':
            criterion = [criterion.cuda(), PatchMixLoss(criterion=criterion).cuda()]
    elif args.method == 'patchmix_cl':
        criterion = [criterion.cuda(), PatchMixConLoss(temperature=args.temperature).cuda()]

    
    #print("device_count :" , torch.cuda.device_count())
    #if torch.cuda.device_count() > 1:
        #model = torch.nn.DataParallel(model)
        
    model.cuda()
    projector.cuda()
    
    
    
    optim_params = list(model.parameters()) + list(classifier.parameters()) + list(projector.parameters())
    optimizer = set_optimizer(args, optim_params)
    
    return model, classifier, projector, criterion, optimizer



def train(train_loader, model, classifier, projector, criterion, optimizer, epoch, args, scaler=None):
   
    
    model.train()
    classifier.train()
    projector.train()
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    
    grad_norms = AverageMeter()  # Track gradient norms

    end = time.time()
    
    for idx, (images,raw_audio, labels) in enumerate(train_loader):
        try:  # 에러 캐치를 위한 try-except 추가
            # data load
            data_time.update(time.time() - end)
            
            images = images.cuda(non_blocking=True)
            raw_audio = raw_audio.cuda(non_blocking=True)
            
        
            class_labels = labels[0].cuda(non_blocking=True)
            device_labels = labels[1].cuda(non_blocking=True)
            patient_labels = labels[2].cuda(non_blocking=True)
                
            bsz = class_labels.shape[0] 
            

            
            
            if args.ma_update:
                # store the previous iter checkpoint
                with torch.no_grad():
                    ma_ckpt = [deepcopy(model.state_dict()), deepcopy(classifier.state_dict()), deepcopy(projector.state_dict())]
                    lamb = None

            warmup_learning_rate(args, epoch, idx, len(train_loader), optimizer)

            with torch.cuda.amp.autocast():
                if args.method == 'ce':

                    if args.nospec:
                        features = model(images, args=args, alpha=lamb,training=True)
                    else:
                        features = model(args.transforms(images), args=args, alpha=lamb, training=True)
                    
                    
                    output = classifier(features)
                    loss = criterion[0](output, class_labels)

            
                elif args.method == 'proto':
                    
                    if args.nospec:
                        features = model(images,raw_audio)
                    else:
                        features = model(args.transforms(images),raw_audio)
                    
                    # # Debug features
                    # if idx % args.print_freq == 0:
                    #     debug_info("Features", features)
                    
                    
                    output = classifier(features)
                    
                    
                    # # Debug output and loss calculation
                    # if idx % args.print_freq == 0:
                    #     debug_info("Model output", output)
                    #     print(f"Target labels: min={class_labels.min().item()}, max={class_labels.max().item()}")
                    
                    
                    loss = criterion[0](output, class_labels)
                    
                    # # Debug loss
                    # if idx % args.print_freq == 0:
                    #     print(f"Loss value: {loss.item():.4f}")
                    #     if torch.isnan(loss).any():
                    #         print("WARNING: NaN detected in loss!")
                    #         print("Output probabilities:", torch.softmax(output, dim=1))
                                


            losses.update(loss.item(), bsz)
            

            [acc1], _ = accuracy(output, class_labels, topk=(1,))


            top1.update(acc1[0], bsz)

            optimizer.zero_grad()
        
            scaler.scale(loss).backward()
            
            # # Debug gradients
            # if idx % args.print_freq == 0:
            #     print("\nGradient Analysis:")
            #     total_norm = 0
            #     for name, param in model.named_parameters():
            #         if param.grad is not None:
            #             param_norm = param.grad.data.norm(2)
            #             total_norm += param_norm.item() ** 2
            #             if torch.isnan(param.grad).any():
            #                 print(f"NaN gradient detected in {name}")
            #                 print(f"Parameter stats - min: {param.min().item():.4f}, max: {param.max().item():.4f}, mean: {param.mean().item():.4f}")
            #                 print(f"Gradient stats - min: {param.grad.min().item():.4f}, max: {param.grad.max().item():.4f}, mean: {param.grad.mean().item():.4f}")
                
            #     total_norm = total_norm ** 0.5
            #     grad_norms.update(total_norm, bsz)
            #     print(f"Total gradient norm: {total_norm:.4f}")
            
            scaler.step(optimizer)
            scaler.update()
            
            # # Optimizer step with debugging
            # try:
            #     scaler.step(optimizer)
            #     scaler.update()
            # except RuntimeError as e:
            #     print(f"Optimizer step failed: {str(e)}")
            #     print("Current learning rate:", optimizer.param_groups[0]['lr'])
            #     raise e

        
    
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if args.ma_update:
                with torch.no_grad():
                    # exponential moving average update
                    model = update_moving_average(args.ma_beta, model, ma_ckpt[0])
                    classifier = update_moving_average(args.ma_beta, classifier, ma_ckpt[1])
                    projector = update_moving_average(args.ma_beta, projector, ma_ckpt[2])

            # print info
            if (idx + 1) % args.print_freq == 0:
                print('Train: [{0}][{1}/{2}]\t'
                    'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                    'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                    epoch, idx + 1, len(train_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses, top1=top1))
                sys.stdout.flush()

        except Exception as e:
                print(f"Error in training batch {idx}: {str(e)}")
                print("Current batch statistics:")
                print(f"Images shape: {images.shape}")
                print(f"Labels shape: {class_labels.shape}")
                print(f"Device labels shape: {device_labels.shape}")
                print(f"Patient labels shape: {patient_labels.shape}")
                raise e
        
        
    # debugger.remove_hooks()
    return losses.avg, top1.avg



def plot_and_save_roc_curve(true_labels, predicted_probs, save_path):
    fpr, tpr, _ = roc_curve(true_labels, predicted_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(save_path)
    plt.close()
    
    
def validate(val_loader, model, classifier, criterion, args, best_acc, best_model=None):
    save_bool = False
    model.eval()
    classifier.eval()
    
    all_labels = []
    all_preds = []
    all_patient_preds = []
    all_patient_labels = []
    
    pre_model_features = []
    post_model_features = []
    all_devices = []
    all_patients = []
    all_classes = []
    all_predicted_classes = []
    
    patient_performance = {}
    
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    hits, counts = [0.0] * args.n_cls, [0.0] * args.n_cls

    with torch.no_grad():
        end = time.time()
        for idx, (images,raw_audio, labels) in enumerate(val_loader):
            images = images.cuda(non_blocking=True)
            
            if args.visualize_embeddings or args.confusion_matrix:
               class_labels = labels[0].cuda(non_blocking=True)
               device_labels = labels[1].cuda(non_blocking=True)
               patient_labels = labels[2].cuda(non_blocking=True)
               labels = class_labels
               
               if args.visualize_embeddings:
                   pre_model_features.append(images.cpu().view(images.size(0), -1).numpy())
                   all_devices.extend(device_labels.cpu().numpy())
                   all_patients.extend(patient_labels.cpu().numpy())
                   all_classes.extend(class_labels.cpu().numpy())
            else:
                labels = labels.cuda(non_blocking=True)

            bsz = labels.shape[0]

            # Forward pass based on method
            if args.method == 'proto':
                features = model(images,raw_audio) 
                
                output = classifier(features)
                loss = criterion[0](output, labels)
                if args.visualize_embeddings:
                    post_model_features.append(features.cpu().numpy())
                
                

            else:
                # 기존 AST 처리
                features = model(images, args=args, training=False)
                output = classifier(features)
                loss = criterion[0](output, labels)
                if args.visualize_embeddings:
                    post_model_features.append(features.cpu().numpy())
                
                
                
            _, preds = torch.max(output, 1)

            # Common processing
            losses.update(loss.item(), bsz)
      
            [acc1], _ = accuracy(output, labels, topk=(1,))
            top1.update(acc1[0], bsz)
            
            all_predicted_classes.extend(preds.cpu().numpy())
            
            # Update confusion matrix data
            if args.confusion_matrix:
                for i in range(len(patient_labels)):
                    patient_id = patient_labels[i].item()
                    is_correct = preds[i].item() == labels[i].item()
                    
                    if patient_id not in patient_performance:
                        patient_performance[patient_id] = {'correct': 0, 'incorrect': 0}
                    
                    if is_correct:
                        patient_performance[patient_id]['correct'] += 1
                    else:
                        patient_performance[patient_id]['incorrect'] += 1

            # Update hits and counts
            for idx in range(preds.shape[0]):
                counts[labels[idx].item()] += 1.0
                if not args.two_cls_eval:
                    if preds[idx].item() == labels[idx].item():
                        hits[labels[idx].item()] += 1.0
                else:
                    if labels[idx].item() == 0 and preds[idx].item() == labels[idx].item():
                        hits[labels[idx].item()] += 1.0
                    elif labels[idx].item() != 0 and preds[idx].item() > 0:
                        hits[labels[idx].item()] += 1.0

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

            # Calculate metrics
            sp, se, sc, f1_normal = get_score(hits, counts)

            batch_time.update(time.time() - end)
            end = time.time()

            # Print progress
            if (idx + 1) % args.print_freq == 0:
                print_str = ('Test: [{0}/{1}]\t'
                           'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                           'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                           'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                           'S_p {sp:.3f}\tS_e {se:.3f}\tScore {sc:.3f}\t'
                           'F1 {f1:.3f}'.format(
                            idx + 1, len(val_loader), batch_time=batch_time,
                            loss=losses, top1=top1, sp=sp, se=se, sc=sc,
                            f1=f1_normal))
            
                print(print_str)

    # Visualization and analysis
    if args.visualize_embeddings:
        visualize_tsne(pre_model_features, post_model_features, 
                      all_devices, all_patients, all_classes, 
                      all_predicted_classes, args, prefix='test')

    # Confusion matrices
    if args.confusion_matrix:
        # Class prediction confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.savefig(os.path.join(args.save_folder, 'confusion_matrix.png'))
        plt.close()
        
        # Patient prediction confusion matrix (only for pairwise_ast)
        if args.method == 'pairwise_ast':
            cm_patient = confusion_matrix(all_patient_labels, all_patient_preds)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm_patient, annot=True, fmt='d', cmap='Blues',
                       xticklabels=['Different', 'Same'],
                       yticklabels=['Different', 'Same'])
            plt.title('Patient Prediction Confusion Matrix')
            plt.savefig(os.path.join(args.save_folder, 'patient_confusion_matrix.png'))
            plt.close()
        
        # Patient performance analysis
        patient_analysis(patient_performance, args.save_folder)

    # Save best model

    if sc > best_acc[-2] and se > 5:
        save_bool = True
        best_acc = [sp, se, sc, f1_normal]
        best_model = [deepcopy(model.state_dict()), deepcopy(classifier.state_dict())]
        
    print(' * S_p: {:.2f}, S_e: {:.2f}, Score: {:.2f} (Best S_p: {:.2f}, S_e: {:.2f}, Score: {:.2f})'.format(sp, se, sc, best_acc[0], best_acc[1], best_acc[2]))
    print(' * F1 Score: {:.2f} (F1 Score: {:.2f})'.format(f1_normal, best_acc[3]))
    print(' * Acc@1 {top1.avg:.2f}'.format(top1=top1))
    
    
    return best_acc, best_model, save_bool

def patient_analysis(patient_performance, save_folder):
    """Patient performance analysis and visualization"""
    patient_ids = sorted(patient_performance.keys())
    matrix_data = []
    
    for patient_id in patient_ids:
        correct = patient_performance[patient_id]['correct']
        incorrect = patient_performance[patient_id]['incorrect']
        total = correct + incorrect
        accuracy_patient = correct / total if total > 0 else 0
        matrix_data.append([patient_id, correct, incorrect, total, accuracy_patient])

    df = pd.DataFrame(matrix_data, 
                     columns=['Patient ID', 'Correct', 'Incorrect', 'Total', 'Accuracy'])
    
    plt.figure(figsize=(12, len(patient_ids) * 0.5 + 2))
    sns.heatmap(df[['Correct', 'Incorrect']], annot=True, fmt='d', cmap='RdYlGn',
                xticklabels=['Correct', 'Incorrect'], yticklabels=df['Patient ID'])
    plt.title('Patient Performance Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, 'patient_performance_matrix.png'))
    plt.close()

    df.to_csv(os.path.join(save_folder, 'patient_performance_matrix.csv'), index=False)

def print_final_results(method, sp, se, sc, f1_normal, best_acc, top1, patient_acc=None):
    """Print final validation results"""
    if method == 'pairwise_ast':
        print('Final Results:')
        print(' * S_p: {:.2f}, S_e: {:.2f}, Score: {:.2f}, Patient Acc: {:.2f}'.format(
            sp, se, sc, patient_acc*100))
        print(' * Best - S_p: {:.2f}, S_e: {:.2f}, Score: {:.2f}, Patient Acc: {:.2f}'.format(
            best_acc[0], best_acc[1], best_acc[2], best_acc[4]*100))
    else:
        print(' * S_p: {:.2f}, S_e: {:.2f}, Score: {:.2f} (Best S_p: {:.2f}, S_e: {:.2f}, Score: {:.2f})'.format(
            sp, se, sc, best_acc[0], best_acc[1], best_acc[2]))
    
    print(' * F1 Score: {:.2f} (Best: {:.2f})'.format(f1_normal, best_acc[3]))
    print(' * Acc@1 {top1.avg:.2f}'.format(top1=top1))



def visualize_tsne(pre_features, post_features, devices, patients, classes, predicted_classes, args, prefix=''):
    print('Performing t-SNE...')
    tsne = TSNE(n_components=2, random_state=42, n_jobs=-1)
    
    plt.rcParams.update({'font.size': 36})  # 기본 폰트 크기 설정
    
    for feature_type, features in [('pre', pre_features), ('post', post_features)]:
        print(f"Shape of {feature_type}_features: {features.shape}")
        reduced_features = tsne.fit_transform(features)
        print(f"Shape of reduced_{feature_type}_features: {reduced_features.shape}")

        # 환자 색상 설정
        highlight_patients = [101, 223, 149, 174, 156]
        colors = plt.cm.rainbow(np.linspace(0, 1, len(highlight_patients)))
        color_dict = dict(zip(highlight_patients, colors))
        patient_colors = ['gray' if p not in highlight_patients else color_dict[p] for p in patients]

        # 시각화
        plt.figure(figsize=(24, 20))  # 그림 크기 증가
        plt.scatter(reduced_features[:, 0], reduced_features[:, 1], 
                    c=patient_colors, 
                    marker='o', alpha=0.7, s=300)
        
        # 범례 추가
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label=f'Patient {l}',
                           markerfacecolor=color_dict[l], markersize=25) for l in highlight_patients]
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', label='Other Patients',
                               markerfacecolor='gray', markersize=25))
        
        plt.legend(handles=legend_elements, loc='best', ncol=2,  fontsize=32, 
                   bbox_transform=plt.gcf().transFigure, borderaxespad=0)
        plt.title(f'{prefix} {feature_type}-model t-SNE visualization by patient', fontsize=42)
        plt.tight_layout()
        plt.savefig(os.path.join(args.save_folder, f'{prefix}_{feature_type}_tsne_patient.png'), dpi=300)
        plt.close()

    print('t-SNE visualization completed.')







def main():
    args = parse_args()
    with open(os.path.join(args.save_folder, 'train_args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)

    # fix seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    cudnn.deterministic = True
    cudnn.benchmark = True
    
    best_model = None


    best_acc = [0, 0, 0, 0]  # Specificity, Sensitivity, Score, F1
    
    
    if not args.nospec:
        args.transforms = SpecAugment(args)

        
    train_loader, val_loader, args = set_loader(args)

    model, classifier,projector, criterion, optimizer = set_model(args)
        
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch'] 
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            args.start_epoch += 1
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    else:
        args.start_epoch = 1

    # use mix_precision:
    scaler = torch.cuda.amp.GradScaler()
    
    print('*' * 20)
    print('Checkpoint Name: {}'.format(args.model_name))
     
    if not args.eval:
        print('Training for {} epochs on {} dataset'.format(args.epochs, args.dataset))
        for epoch in range(args.start_epoch, args.epochs+1):
        
            
            adjust_learning_rate(args, optimizer, epoch)

            # train for one epoch
            time1 = time.time()
            
            loss, acc = train(train_loader, model, classifier,projector, criterion, optimizer, epoch, args, scaler)
            time2 = time.time()
            print('Train epoch {}, total time {:.2f}, accuracy:{:.2f}'.format(epoch, time2-time1, acc))
            
            # eval for one epoch
            best_acc, best_model, save_bool = validate(val_loader, model, classifier, criterion, args, best_acc, best_model)
            
            # save a checkpoint of model and classifier when the best score is updated
            if save_bool:            
                save_file = os.path.join(args.save_folder, 'best_epoch_{}.pth'.format(epoch))
                print('Best ckpt is modified with Score = {:.2f} when Epoch = {}'.format(best_acc[2], epoch))
                # print('Best ckpt is modified with F1 = {:.2f} when Epoch = {}'.format(best_acc[3], epoch))
                save_model(model, optimizer, args, epoch, save_file, classifier)
                
                        
            if epoch % args.save_freq == 0:
                save_file = os.path.join(args.save_folder, 'epoch_{}.pth'.format(epoch))
                save_model(model, optimizer, args, epoch, save_file, classifier)
            

        # save a checkpoint of classifier with the best accuracy or score
        save_file = os.path.join(args.save_folder, 'best.pth')
        model.load_state_dict(best_model[0])
        classifier.load_state_dict(best_model[1])
        save_model(model, optimizer, args, epoch, save_file, classifier)
        
    else:
        print("correct")
        print('Testing the pretrained checkpoint on {} dataset'.format(args.dataset))
        best_acc, _, _  = validate(val_loader, model, classifier, criterion, args, best_acc)
        model.eval()  # Set the model to evaluation mode
        

   
    update_json('%s' % args.model_name, best_acc, path=os.path.join(args.save_dir, 'results.json'))
    print('Checkpoint {} finished'.format(args.model_name))
    
if __name__ == '__main__':
    main()

