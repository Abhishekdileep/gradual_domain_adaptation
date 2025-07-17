import torch 
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import  Dataset ,DataLoader , TensorDataset
import os 
PATH = ""

label_keys = { 'airport' : 0 , 'shopping_mall' : 1 , 'park' : 2, 'street_pedestrian' : 3,
       'street_traffic' : 4 , 'metro_station' : 5 , 'public_square' : 6 , 'metro' : 7, 'bus' : 8 ,
       'tram' : 9}

class AudioDataset(Dataset):
    """
    A custom PyTorch Dataset for handling audio files using torchaudio.

    This class takes file paths and labels, loads the audio files,
    and preprocesses them by resampling, converting to mono, and
    padding/truncating to a fixed length.
    """
    def __init__(self, x_col, y_col, mel ,  target_sr=44100, target_duration_sec=10, path=PATH ,label_keys = label_keys ):
        if len(x_col) != len(y_col):
            raise ValueError("Input columns X (filenames) and Y (labels) must have the same length.")

        self.file_paths = x_col
        self.labels = y_col
        self.target_sr = target_sr
        self.num_samples = int(target_sr * target_duration_sec)
        self.mel = mel
        self.dir_path = path
        self.label_keys = label_keys
    def __len__(self):
        return len(self.file_paths)

    def OneHot(self, label, num_classes):
        one_hot = torch.zeros(num_classes, dtype=torch.float32)
        one_hot[label] = 1.0
        return one_hot

    def __getitem__(self, index):
        
        audio_path = self.file_paths.iloc[index]
        label = self.labels.iloc[index]
        labelkey = self.label_keys[label]
        try:
    
            # waveform, sr = lr.load(audio_path, sr=self.target_sr)
            waveform, sr = torchaudio.load(os.path.join(self.dir_path , audio_path))
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
            # labelkey = self.OneHot(labelkey, num_classes=len(label_keys.keys()))  # Convert label to one-hot encoding
            spectrogram = self.mel(waveform) 
            return spectrogram, labelkey

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
    def __init__(self, x_col_src, x_col_tgt, y_col, mel , target_sr=32000, target_duration_sec=10, path=PATH , label_keys = label_keys ):
        if len(x_col_src) != len(y_col):
            raise ValueError("Input columns X (filenames) and Y (labels) must have the same length.")

        self.file_path_src = x_col_src
        self.file_path_tgt = x_col_tgt
        self.labels = y_col
        self.target_sr = target_sr
        self.num_samples = int(target_sr * target_duration_sec)
        self.alpha =[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
        self.mel = mel
        self.path = PATH 
        self.label_keys = label_keys

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
        labelkey = self.label_keys[label]
        try:

            waveform1, sr1 = torchaudio.load(os.path.join(self.dir_path ,audio_path_src))
            waveform2, sr2 = torchaudio.load(os.path.join(self.dir_path ,audio_path_tgt))
   
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

            spectrogram_mixed = []
            for alpha in self.alpha:
                spectrogram = self.mel(waveform1, waveform2, alpha)
                spectrogram_mixed.append(spectrogram)
            # spectrogram_mixed = torch.cat(spectrogram_mixed,axis=0)
            spectrogram_mixed = torch.stack(spectrogram_mixed,axis=1)
            spectrogram_mixed = spectrogram_mixed.squeeze(0)
            # spectrogram_mixed = spectrogram_mixed.reshape(spectrogram_mixed.shape[1],spectrogram_mixed.shape[0],spectrogram_mixed.shape[2],spectrogram_mixed.shape[3])
            return spectrogram_mixed, labelkey

        except Exception as e:
            print(f"Error loading or processing file: {audio_path_src}")
            print(f"Error: {e}")
            # Return a dummy tensor and a placeholder label if an error occurs
            return torch.zeros(self.num_samples), -1
