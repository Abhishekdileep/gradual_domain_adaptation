import os
import pandas as pd
import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader
from scipy.io.wavfile import write as write_wav # For creating dummy files
from dataset_passt import AugmentMelSTFT2
from hear21passt.models.preprocess import AugmentMelSTFT
import ipdb
import librosa as lr 
import warnings
warnings.filterwarnings("ignore")

PATH = '/home/abhishek/Code/M2D/Dcase-2020/TAU-urban-acoustic-scenes-2020-mobile-development'

label_keys = { 'airport' : 0 , 'shopping_mall' : 1 , 'park' : 2, 'street_pedestrian' : 3,
       'street_traffic' : 4 , 'metro_station' : 5 , 'public_square' : 6 , 'metro' : 7, 'bus' : 8 ,
       'tram' : 9}

train_src = ['a']
train_tgt = ['b']
test_src = ['a']
test_tgt = ['b']

class AudioDataset(Dataset):
    """
    A custom PyTorch Dataset for handling audio files using torchaudio.

    This class takes file paths and labels, loads the audio files,
    and preprocesses them by resampling, converting to mono, and
    padding/truncating to a fixed length.
    """
    def __init__(self, x_col, y_col, mel ,  target_sr=44100, target_duration_sec=10):
        if len(x_col) != len(y_col):
            raise ValueError("Input columns X (filenames) and Y (labels) must have the same length.")

        self.file_paths = x_col
        self.labels = y_col
        self.target_sr = target_sr
        self.num_samples = int(target_sr * target_duration_sec)
        self.mel = mel
    def __len__(self):
        return len(self.file_paths)

    def OneHot(self, label, num_classes):
        one_hot = torch.zeros(num_classes, dtype=torch.float32)
        one_hot[label] = 1.0
        return one_hot

    def __getitem__(self, index):
        
        audio_path = self.file_paths.iloc[index]
        label = self.labels.iloc[index]

        try:
            waveform, sr = lr.load(audio_path, sr=self.target_sr)
            waveform, sr = torchaudio.load(audio_path)
            if sr != self.target_sr:
                resampler = T.Resample(orig_freq=sr, new_freq=self.target_sr)
                waveform = resampler(waveform)
            current_samples = waveform.shape[1]
            if current_samples > self.num_samples:
                # Truncate
                waveform = waveform[:self.num_samples]
            elif current_samples < self.num_samples:
                # Pad with zeros
                padding_needed = self.num_samples - current_samples
                # torch.pad takes (padding_left, padding_right)
                waveform = torch.nn.functional.pad(waveform, (0, padding_needed))
            label = self.OneHot(label, num_classes=len(self.labels.unique()))  # Convert label to one-hot encoding
            spectrogram = self.mel(waveform) 
            return spectrogram, label

        except Exception as e:
            print(f"Error loading or processing file: {audio_path}")
            print(f"Error: {e}")
            # Return a dummy tensor and a placeholder label if an error occurs
            return torch.zeros(self.num_samples), -1

class AudioDataset2(Dataset):
    """
    A custom PyTorch Dataset for handling audio files using torchaudio.

    This class takes file paths and labels, loads the audio files,
    and preprocesses them by resampling, converting to mono, and
    padding/truncating to a fixed length.
    """
    def __init__(self, x_col_src, x_col_tgt, y_col, mel , target_sr=32000, target_duration_sec=10):
        if len(x_col_src) != len(y_col):
            raise ValueError("Input columns X (filenames) and Y (labels) must have the same length.")

        self.file_path_src = x_col_src
        self.file_path_tgt = x_col_tgt
        self.labels = y_col
        self.target_sr = target_sr
        self.num_samples = int(target_sr * target_duration_sec)
        self.alpha =[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
        self.mel = mel

    def __len__(self):
        return len(self.file_path_src)

    def OneHot(self, label, num_classes):
        one_hot = torch.zeros(num_classes, dtype=torch.float32)
        one_hot[label] = 1.0
        return one_hot

    def __getitem__(self, index):
        audio_path_src = self.file_path_src[index]
        audio_path_tgt = self.file_path_tgt[index]
        label = self.labels[index]
        try:

            waveform1, sr1 = torchaudio.load(audio_path_src)
            waveform2, sr2 = torchaudio.load(audio_path_tgt)
   

            if sr1 != self.target_sr:
                resampler = T.Resample(orig_freq=sr1, new_freq=self.target_sr)
                waveform1 = resampler(waveform1)
            if sr2 != self.target_sr:
                resampler = T.Resample(orig_freq=sr2, new_freq=self.target_sr)
                waveform2 = resampler(waveform2)
            # Squeeze to remove the channel dimension, making it (num_samples,)
            current_samples = waveform1.shape[1]
            if current_samples > self.num_samples:
                # Truncate
                waveform1 = waveform1[:self.num_samples]
                waveform2 = waveform2[:self.num_samples]
            elif current_samples < self.num_samples:
                # Pad with zeros
                padding_needed = self.num_samples - current_samples
                waveform1 = torch.nn.functional.pad(waveform1, (0, padding_needed))
                waveform2 = torch.nn.functional.pad(waveform2, (0, padding_needed))
            label = self.OneHot(label, num_classes=len(self.labels.unique()))  
            spectrogram_mixed = []
            for alpha in self.alpha:
                spectrogram = self.mel(waveform1, waveform2, alpha)
                spectrogram_mixed.append(spectrogram)
            # spectrogram_mixed = torch.cat(spectrogram_mixed,axis=0)
            spectrogram_mixed = torch.stack(spectrogram_mixed,axis=1)
            spectrogram_mixed = spectrogram_mixed.squeeze(0)
            # spectrogram_mixed = spectrogram_mixed.reshape(spectrogram_mixed.shape[1],spectrogram_mixed.shape[0],spectrogram_mixed.shape[2],spectrogram_mixed.shape[3])
            return spectrogram_mixed, label

        except Exception as e:
            print(f"Error loading or processing file: {audio_path_src}")
            print(f"Error: {e}")
            # Return a dummy tensor and a placeholder label if an error occurs
            return torch.zeros(self.num_samples), -1

def Passt_with_mix(data_path=PATH):
    #######################
    ### Load CSV files ####
    #######################
    df = pd.read_csv(os.path.join(data_path, 'meta.csv'),sep='\t')
    evaluate_df = pd.read_csv(os.path.join(data_path, 'fold1_evaluate.csv'), sep='\t')
    train_df = pd.read_csv(os.path.join(data_path, 'fold1_train.csv'), sep='\t')
    # test_df = pd.read_csv(os.path.join(data_path, 'fold1_test.csv') , sep='\t')
    df['file_number'] = df['filename'].apply(lambda x: x.split('-')[2]  +  x.split('-')[3])

    
    ##############################################
    ### Split according to Dcase train-test ######
    ##############################################
    
    train_with_meta = pd.merge(train_df, df[['filename', 'scene_label', 'source_label' , 'file_number']], on=['filename', 'scene_label'], how='left')
    evaluate_with_meta = pd.merge(evaluate_df, df[['filename', 'scene_label', 'source_label' , 'file_number']], on=['filename', 'scene_label'], how='left')

    train_with_meta = pd.merge(train_df, df[['filename', 'scene_label', 'source_label' , 'file_number']], on=['filename', 'scene_label'], how='left')
    evaluate_with_meta = pd.merge(evaluate_df, df[['filename', 'scene_label', 'source_label' , 'file_number']], on=['filename', 'scene_label'], how='left')
    #########################
    # Define domain groups ##
    #########################
    train_src = ['a']
    train_tgt = ['b']
    test_src = ['a']
    test_tgt = ['b']
    # original dcase dataset 
    # train_domain = ['a' , 'b', 'c', 's1', 's2', 's3']
    # test_domain = ['a', 'b', 'c', 's1', 's2', 's3', 's4', 's5', 's6' ]
    
    print("Train csv with meta information of length" , len(train_with_meta))
    print("Evaluate with meta information of length" , len(evaluate_with_meta))


    ###########################################
    ### Create Source and Target Dataframes ###
    ###########################################
    
    
    train_src_df = {d: train_with_meta[train_with_meta['source_label'] == d].copy() for d in train_src}
    train_tgt_df = {d: train_with_meta[train_with_meta['source_label'] == d].copy() for d in train_tgt}

    test_src_df = {d: evaluate_with_meta[evaluate_with_meta['source_label'] == d].copy() for d in test_src}
    test_tgt_df = {d: evaluate_with_meta[evaluate_with_meta['source_label'] == d].copy() for d in test_tgt}

    ##############################################
    ## Mixed up devices that are paried segment ##
    ##############################################
    src_devices_mixup = []
    target_devices_mixup = []  
    filtered_df = train_with_meta[train_with_meta['source_label'].isin(['a', 'b'])]
    label_list = [] 
    for i in filtered_df.groupby('file_number'):
        if len(i[1]) > 1 :
            label = i[1]['scene_label'].iloc[0]
            labelkey = label_keys[label]
            label_list.append(labelkey) 
            for j in i[1].iterrows() : 
                device = j[1]['source_label']
                
                if device in train_src : ### Device that is source 
                    src_devices_mixup.append(os.path.join(PATH , j[1]['filename']) )
                elif device in train_tgt : 
                    target_devices_mixup.append(os.path.join(PATH , j[1]['filename']) )
    print("Paired devices found: ", len(src_devices_mixup))
    
    return src_devices_mixup , target_devices_mixup , label_list , train_src_df, train_tgt_df, test_src_df, test_tgt_df

if __name__ == '__main__':


    src_devices_mixup , target_devices_mixup, label_list , train_src_df, train_tgt_df, test_src_df, test_tgt_df = Passt_with_mix()
    src_df = pd.DataFrame({
        'filename': src_devices_mixup,
        'label': label_list
    })
    
    tgt_df = pd.DataFrame({
        'filename': target_devices_mixup,
        'label': label_list
    })

    # this all for paired
    target_sr = 32000
    target_duration = 10
    mel2 = AugmentMelSTFT2(n_mels=128, sr=32000, win_length=800, hopsize=320, n_fft=1024, freqm=48,
                         timem=192,
                         htk=False, fmin=0.0, fmax=None, norm=1, fmin_aug_range=10,
                         fmax_aug_range=2000)

    
    # mixup_dataset = AudioDataset2(
    #     x_col_src=src_df['filename'],
    #     x_col_tgt=tgt_df['filename'],
    #     y_col=tgt_df['label'],  
    #     mel = mel2,
    #     target_sr=target_sr,
    #     target_duration_sec=target_duration
    # )

    # mixup_loader = DataLoader(
    #     dataset=mixup_dataset,
    #     batch_size=4,
    #     shuffle=True
    # )
    ipdb.set_trace()
    
    mel = AugmentMelSTFT(n_mels=128, sr=32000, win_length=800, hopsize=320, n_fft=1024, freqm=48,
                         timem=192,
                         htk=False, fmin=0.0, fmax=None, norm=1, fmin_aug_range=10,
                         fmax_aug_range=2000)
    # i dont have all the src and tgt that is why this new one 
    train_src_df = train_src_df['a']
    singluar_src_dataset = AudioDataset(
        x_col=train_src_df['filename'],
        y_col=train_src_df['scene_label'],  
        mel = mel,
        target_sr=target_sr,
        target_duration_sec=target_duration
    )

    # Get one batch of data
    first_batch_waveforms, first_batch_labels = next(iter(mixup_loader))
    #Very neccesary to reshape it like this 
    first_batch_waveforms = first_batch_waveforms.reshape(first_batch_waveforms.shape[0] * first_batch_waveforms.shape[1] , first_batch_waveforms.shape[2] , first_batch_waveforms.shape[3]) 

    print("Retrieved one batch from the DataLoader:")
    print(f"Waveforms batch shape: {first_batch_waveforms.shape}")
    print(f"Labels batch shape: {first_batch_labels.shape}")
    print(f"Labels in batch: {first_batch_labels}")
    
    
import torch
from hear21passt.base import get_basic_model, get_model_passt
from hear21passt.models.preprocess import AugmentMelSTFT
from dataset_passt import AugmentMelSTFT2
from wrapper import PasstBasicWrapper2
# high-res pre-trained on Audioset
net = get_model_passt(arch="passt_s_swa_p16_128_ap476")

# hopsize=160 for this pretrained model
model = PasstBasicWrapper2(net=net, mode="embed_only")
model = model.cuda().to('cuda:1')
model.eval()
seconds = 10
alpha = 0.5
mel = mel.cuda()
with torch.no_grad():
    # import ipdb
    # ipdb.set_trace()
    first_batch_waveforms = first_batch_waveforms.unsqueeze(1)
    first_batch_waveforms = first_batch_waveforms.to('cuda:1')
    torch.save(first_batch_waveforms , 'saved_files/first_wave.pt')
    output = model(first_batch_waveforms)
    print(f"Final output : {output.shape}")  # should be (3, 128, 192)
    # The shape should be torch.Size([batch_size, num_samples]), e.g., [2, 64000]
    # assert list(first_batch_waveforms.shape) == [2, target_sr * target_duration]
