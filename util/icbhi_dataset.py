from curses import meta
import os
import pickle
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from glob import glob
import scipy.signal as signal

import torch
from torch.utils.data import Dataset
from copy import deepcopy
from PIL import Image
import librosa

from .icbhi_util import get_annotations, generate_fbank, get_individual_cycles_torchaudio, cut_pad_sample_torchaudio,generate_spectrogram
from .augmentation import augment_raw_audio
import torchaudio
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt



def get_icbhi_device_infor(icbhi_device, args):
    
    if args.device_mode == 'none':
        device_label = -1
    
    
    elif args.device_mode == 'mixed':
        if icbhi_device == 'L':
            device_label = 0
        elif icbhi_device == 'A':
            device_label = 1
        elif icbhi_device == 'M':
            device_label = 2
        else:
            device_label = 3
    
                   
    return device_label




class ICBHIDataset(Dataset):
    def __init__(self, train_flag, transform, args, print_flag=True):
        train_data_folder = os.path.join(args.data_folder, 'training', 'real')
        test_data_folder = os.path.join(args.data_folder, 'test', 'real')
        self.train_flag = train_flag
        
        self.frame_shift = 10
        
    
            
        if self.train_flag:
            self.data_folder = train_data_folder
        else:
            self.data_folder = test_data_folder
            
        self.transform = transform
        self.args = args
        
        # 추가된 속성: 환자별 라벨 저장
        self.patient_labels = []
    
        self.targets = []
        
        # parameters for spectrograms
        self.sample_rate = self.args.sample_rate
        self.n_mels = self.args.n_mels        
        
        self.class_nums = np.zeros(args.n_cls)
        
        self.device_nums = np.zeros(4)
        
        self.device_class_nums = np.zeros((4, args.n_cls))  # 새로운 속성 추가
        
        self.data_glob = sorted(glob(self.data_folder+'/*.wav'))
        
        print('Total length of dataset is', len(self.data_glob))
        
        
        # patient ID 매핑을 위한 dictionary 생성
        unique_patients = sorted(set([file_id_tmp.split('_')[0] for file_id_tmp in [os.path.split(index)[1].split('.wav')[0] for index in self.data_glob]]))
        self.patient_to_idx = {pid: idx for idx, pid in enumerate(unique_patients)}
        self.num_patients = len(unique_patients)
        
        if print_flag:
            print(f"Number of unique patients: {self.num_patients}")
            
                                    
        # ==========================================================================
        """ convert fbank """
        self.audio_images = []
        self.raw_audio = []

        for index in self.data_glob: #for the training set, 4142
            _, file_id = os.path.split(index)
            
            
            
            audio, sr = torchaudio.load(index)
    
            
            
            file_id_tmp = file_id.split('.wav')[0]

            
            
            # for ICBHI
            label = file_id_tmp.split('_')[-1]
            device_label = file_id_tmp.split('_')[4]
            patient_label = file_id_tmp.split('_')[0]

            
            patient_label = self.patient_to_idx[patient_label]

            # print(label)
            if args.n_cls == 2:
                if label != '0':
                    label = '1'


            self.patient_labels.append(int(patient_label))  # 환자 ID 추가
            self.targets.append(int(label))  
            
            
            self.class_nums[int(label)] += 1
             
                
            
            # for ICBHI
            if device_label == 'LittC2SE':
                icbhi_device = 'L'
            elif device_label == 'Meditron':
                icbhi_device = 'M'
            elif device_label == 'AKGC417L':
                icbhi_device = 'A'
            else:
                icbhi_device = '3200'
            

            
            
            
            # for ICBHI
            device_label = get_icbhi_device_infor(icbhi_device,self.args)
            
            audio, sr = torchaudio.load(index)
            
            self.raw_audio.append(audio) 
            
            
            self.device_nums[int(device_label)] += 1
          
            self.device_class_nums[int(device_label)][int(label)] += 1  # 디바이스별 클래스 수 카운트
            
        
            
            
            audio_image = []
            for aug_idx in range(self.args.raw_augment+1):
                if aug_idx > 0:
                    if self.train_flag:
                        audio = augment_raw_audio(np.asarray(audio.squeeze(0)), self.sample_rate, self.args)
                        audio = cut_pad_sample_torchaudio(torch.tensor(audio), self.args)                
                    
                    image = generate_fbank(self.args, audio, self.sample_rate, n_mels=self.n_mels)


                    audio_image.append(image)
                else:
                    image = generate_fbank(self.args, audio, self.sample_rate, n_mels=self.n_mels,frame_shift=self.frame_shift) 
                    
                    audio_image.append(image)
            
            if self.args.device_mode =='none':
                self.audio_images.append((audio_image, int(label)))
            else:
                self.audio_images.append((audio_image,self.raw_audio, int(label), device_label,int(patient_label)))
                
                 
                
        self.class_ratio = self.class_nums / sum(self.class_nums) * 100
         
        self.device_ratio = self.device_nums / sum(self.device_nums) * 100
         
        if print_flag:
            print('total number of audio data: {}'.format(len(self.data_glob)))
            print('*' * 25)
            print('For the Label Distribution')
            for i, (n, p) in enumerate(zip(self.class_nums, self.class_ratio)):
                print('Class {} {:<9}: {:<4} ({:.1f}%)'.format(i, '('+args.cls_list[i]+')', int(n), p))
            print('For the Device Distribution')
            for i, (n, p) in enumerate(zip(self.device_nums, self.device_ratio)):
                print('Device {} {:<9}: {:<4} ({:.1f}%)'.format(i, '('+args.device_list[i]+')', int(n), p))

            print('\nDetailed Device-Class Distribution:')
            for i in range(4):  # 각 디바이스에 대해
                print(f"Device {i} ({args.device_list[i]}):")
                for j in range(args.n_cls):  # 각 클래스에 대해
                    count = self.device_class_nums[i][j]
                    percentage = (count / self.device_nums[i]) * 100 if self.device_nums[i] > 0 else 0
                    print(f"  Class {j} ({args.cls_list[j]}): {int(count)} ({percentage:.1f}%)")
                print()  # 디바이스 간 빈 줄 추가

        
    
    
    
    def __getitem__(self, index):
        if self.args.device_mode == 'none':
            audio_images, label = self.audio_images[index][0], self.audio_images[index][1]
        else:
            audio_images,raw_audio, label, device_label,patient_label = self.audio_images[index][0], self.audio_images[index][1], self.audio_images[index][2],self.audio_images[index][3],self.audio_images[index][4]

        if self.args.raw_augment and self.train_flag:
            aug_idx = random.randint(0, self.args.raw_augment)
            audio_image = audio_images[aug_idx]
        else:
            audio_image = audio_images[0]

        if self.transform is not None:
            audio_image = self.transform(audio_image)

        
        if self.train_flag:
            if self.args.device_mode == 'none':
                return audio_image, torch.tensor(label)
            else:
                return audio_image,raw_audio, (torch.tensor(label),torch.tensor(device_label),torch.tensor(patient_label))
        else:
            if self.args.visualize_embeddings == True or self.args.confusion_matrix == True:
                return audio_image, (torch.tensor(label),torch.tensor(device_label),torch.tensor(patient_label))
            else:
                return audio_image, torch.tensor(label)
        
        

    def __len__(self):
        return len(self.data_glob)