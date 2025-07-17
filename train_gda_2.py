import torch 
import torch.nn as nn 
from torch.utils.data import  Dataset ,DataLoader , TensorDataset
import torch.optim as optim
import numpy as np
import tqdm
from torchvision.datasets import MNIST
from torchvision.transforms import functional as F
import torch.nn.functional as F1

from scipy.ndimage import rotate
from sklearn.preprocessing import OneHotEncoder
# from dataset_mix_up import rotated_mnist_60_data_func
def rotated_mnist_60_data_func():
	pass
# IMPORTS FOR PASST 
from hear21passt.base import get_basic_model, get_model_passt
from dataset_passt import AugmentMelSTFT2
from wrapper import PasstBasicWrapper2

# IMPORTS FOR Dcase
import pandas as pd
import os 
import torchaudio
import torchaudio.transforms as T
from dataset_passt import AugmentMelSTFT2
from hear21passt.models.preprocess import AugmentMelSTFT
import librosa as lr 
import warnings
import gc
warnings.filterwarnings("ignore")

# =============================================================================
# 1. Pararmeters 
# =============================================================================

BATCH_SIZE = 2
num_runs = 3 
epoch = 50 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
num_classes = 10
source_train_size = 10000
source_val_size = 2000
inter_train_size = 42000
inter_val_size = 46000
target_train_size = 58000
target_val_size = 60000
TEST_STEP_SIZE = 20 # number of steps before the test occurs  
PATH = '/home/abhishek/Code/M2D/Dcase-2020/TAU-urban-acoustic-scenes-2020-mobile-development'
PATH = '/sd/scd/s23103/Dcase-2020/TAU-urban-acoustic-scenes-2020-mobile-development'
SAVE_DIR = 'saved_models'
os.makedirs(SAVE_DIR, exist_ok=True)
# =============================================================================
# 1. Model Definition (PyTorch)
# =============================================================================
class SimpleConvNet(nn.Module):
    """
    A simple Convolutional Neural Network model, equivalent to the likely
    Keras model `simple_softmax_conv_model`.
    """
    def __init__(self, num_classes=10):
        super(SimpleConvNet, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # Reshape input if it's flat
        if x.dim() == 2:
            x = x.view(-1, 28, 28)
        # Add channel dimension if it's missing
        if x.dim() == 3:
            x = x.unsqueeze(1)
        
        out = self.pool1(self.relu1(self.conv1(x)))
        out = self.pool2(self.relu2(self.conv2(out)))
        out = out.view(out.size(0), -1) # Flatten
        out = self.relu3(self.fc1(out))
        out = self.fc2(out)
        return out

class PyTorchConvModel(nn.Module):
    def __init__(self, num_labels, hidden_nodes=32, input_shape=(1, 28, 28)):
        """
        Initializes the layers of the model.

        Args:
            num_labels (int): The number of output classes.
            hidden_nodes (int): The number of filters in the convolutional layers.
            input_shape (tuple): The shape of the input tensor (C, H, W).
                                 Note: PyTorch uses Channels-First format.
        """
        super(PyTorchConvModel, self).__init__()
        
        # Note: Keras uses (H, W, C) while PyTorch uses (C, H, W)
        in_channels = input_shape[0]

        # --- Convolutional Layers ---
        # The Keras 'same' padding with a stride of 2 is approximated here
        # by setting padding=2. This keeps the output dimensions consistent.
        self.conv1 = nn.Conv2d(in_channels, hidden_nodes, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(hidden_nodes, hidden_nodes, kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.Conv2d(hidden_nodes, hidden_nodes, kernel_size=5, stride=2, padding=2)
        
        # --- Regularization and Normalization Layers ---
        self.dropout = nn.Dropout(0.5)
        # BatchNorm2d is used for 2D inputs (i.e., after conv layers)
        self.batch_norm = nn.BatchNorm2d(hidden_nodes)

        # --- Classifier Layers ---
        self.flatten = nn.Flatten()
        
        # In PyTorch, you often need to calculate the input size for the first
        # linear layer manually.
        # Input: 28x28
        # After conv1 (stride 2): ceil(28/2) = 14x14
        # After conv2 (stride 2): ceil(14/2) = 7x7
        # After conv3 (stride 2): ceil(7/2) = 4x4
        # Flattened size = channels * height * width
        flattened_size = hidden_nodes * 4 * 4
        
        self.fc = nn.Linear(flattened_size, num_labels)

    def forward(self, x):
        """Defines the forward pass of the model."""
        # Convolutional block
        x = F1.relu(self.conv1(x))
        x = F1.relu(self.conv2(x))
        x = F1.relu(self.conv3(x))
        
        # Regularization and Normalization
        x = self.dropout(x)
        x = self.batch_norm(x)
        
        # Classifier
        x = self.flatten(x)
        logits = self.fc(x)
        
        # NOTE: The final Softmax activation is NOT applied here.
        # In PyTorch, loss functions like `nn.CrossEntropyLoss` expect raw logits
        # as they combine LogSoftmax and the loss calculation for better
        # numerical stability. You would apply softmax only during inference.
        
        return logits

def simple_softmax_conv_model(num_labels=10, hidden_nodes=32, input_shape=(1 ,28, 28)):
    '''
    Wrapper function similar to the Keras one  
    '''
    return PyTorchConvModel(num_labels, hidden_nodes, input_shape)

class Classifier(nn.Module):
    """
    A simple classifier model that can be used for classification tasks.
    This is a placeholder and should be replaced with the actual classifier model.
    """
    def __init__(self, num_classes=10):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(768, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.thoda_rolu = nn.ReLU(inplace=True)
        self.dropme = nn.Dropout(0.2)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.thoda_rolu(x)
        x = self.dropme(x)
        
        return self.fc2(x)

# class Classifier(nn.Module):
#     """Simplified Classifier"""
#     def __init__(self, input_size=768, num_classes=10):
#         super(Classifier, self).__init__()
#         self.layer = nn.Sequential(
#             nn.Linear(input_size, 128),
#             nn.ReLU(inplace=True),
#             nn.Dropout(0.2),
#             nn.Linear(128, num_classes),
#         )

#     def forward(self, h):
#         return self.layer(h)

class PaSST_wrapper(nn.Module):
    """
    A wrapper for the PaSST model, which is a pre-trained model for audio processing.
    This is a placeholder and should be replaced with the actual PaSST model.
    """
    def __init__(self, num_classes=10):
        super(PaSST_wrapper, self).__init__()
        # Placeholder for the actual PaSST model
        self.net = get_model_passt(arch="passt_s_swa_p16_128_ap476")
        self.model = PasstBasicWrapper2(net=self.net, mode="embed_only")  # Example layer
        self.classifier = Classifier(num_classes)

    def forward(self, x):
        output = self.model(x)
        logits = self.classifier(output)
        return logits 

# =============================================================================
# 2. Dataset Functions (PyTorch)
# =============================================================================

#### Code reused from Tensorflow Dataset , for Controlled Experiments     
def mnist_from_tensorflow():
    (src_train_x , src_train_y , src_val_X , src_val_y , inter_x, inter_y ,  dir_inter_x , dir_inter_y , target_val_X, target_val_y , target_test_X, target_test_y) = rotated_mnist_60_data_func(20000, 22000, 48000, 50000)
    src_train_x = src_train_x.squeeze(3)
    print(src_train_x.shape)  

    src_train_x = np.expand_dims(src_train_x, axis=1 )
    src_val_X = src_val_X.squeeze(3)
    src_val_X = np.expand_dims(src_val_X, axis=1)
    inter_x = inter_x.squeeze(3)
    inter_x = np.expand_dims(inter_x, axis=1)
    dir_inter_x = dir_inter_x.squeeze(3)   
    dir_inter_x = np.expand_dims(dir_inter_x, axis=1)   
    target_val_X = target_val_X.squeeze(3)
    target_val_X = np.expand_dims(target_val_X, axis=1)     
    target_test_X = target_test_X.squeeze(3)
    target_test_X = np.expand_dims(target_test_X , axis=1)  
    print(src_train_x.shape)  

    return (src_train_x , src_train_y , src_val_X , src_val_y , inter_x, inter_y ,  dir_inter_x , dir_inter_y , target_val_X, target_val_y , target_test_X, target_test_y)


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
    def __init__(self, x_col, y_col, mel ,  target_sr=44100, target_duration_sec=10, path=PATH ):
        if len(x_col) != len(y_col):
            raise ValueError("Input columns X (filenames) and Y (labels) must have the same length.")

        self.file_paths = x_col
        self.labels = y_col
        self.target_sr = target_sr
        self.num_samples = int(target_sr * target_duration_sec)
        self.mel = mel
        self.dir_path = path
    def __len__(self):
        return len(self.file_paths)

    def OneHot(self, label, num_classes):
        one_hot = torch.zeros(num_classes, dtype=torch.float32)
        one_hot[label] = 1.0
        return one_hot

    def __getitem__(self, index):
        
        audio_path = self.file_paths.iloc[index]
        label = self.labels.iloc[index]
        labelkey = label_keys[label]
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
    def __init__(self, x_col_src, x_col_tgt, y_col, mel , target_sr=32000, target_duration_sec=10, path=PATH):
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
        labelkey = label_keys[label]
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
            # labelkey = self.OneHot(labelkey, num_classes=len(label_keys.keys()))  
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
    train_tgt = ['b', 'c', 's1', 's2', 's3']
    test_src = ['a']
    test_tgt = ['b', 'c', 's1', 's2', 's3', 's4', 's5', 's6']
    # original dcase dataset 
    # train_domain = ['a' , 'b', 'c', 's1', 's2', 's3']
    # test_domain = ['a', 'b', 'c', 's1', 's2', 's3', 's4', 's5', 's6' ]
    
    print("Train csv with meta information of length" , len(train_with_meta))
    print("Evaluate with meta information of length" , len(evaluate_with_meta))

    train_src = ['a']
    train_tgt = ['b', 'c', 's1', 's2', 's3']
    test_src = ['a']
    test_tgt = ['b', 'c', 's1', 's2', 's3', 's4', 's5', 's6']

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
            
            label_list.append(label) 
            for j in i[1].iterrows() : 
                device = j[1]['source_label']
                
                if device == 'a' : ### Device that is source [use device in [src] or [tgt] ]
                    src_devices_mixup.append(j[1]['filename']) 
                elif device == 'b':
                    target_devices_mixup.append(j[1]['filename']) 
    print("Paired devices found: ", len(src_devices_mixup))
    
    return src_devices_mixup , target_devices_mixup , label_list , train_src_df, train_tgt_df, test_src_df, test_tgt_df


# =============================================================================
# 3. Utility Functions (PyTorch)
# =============================================================================

def self_train_once(student, teacher, source_data ,  unsup_x, confidence_q=0.1, epochs=20):
    """Trains student on hard pseudo-labels from teacher, with confidence filtering."""
    print("  Performing one hard self-training step...")
    
    # ---------- Setting the data loader for unsupervised dataset ---#
    
    unsup_dataset = TensorDataset(torch.FloatTensor(unsup_x), torch.zeros(len(unsup_x))) # dummy labels
    unsup_loader = DataLoader(unsup_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
   # ---------- Predict function for the model  ---#
   
    print('Predicting the Output from source model')
    outputs = [] 
    for data, target in unsup_loader:
        teacher.eval() 
        data , target = data.to(DEVICE) , target.to(DEVICE)
        output = teacher(data)
        outputs.append(output)        
    outputs = torch.cat(outputs,dim=0)            
    
    # ----------- 2. Confidence-based filtering --------------#
    confidence = torch.max(outputs, axis=1).values - torch.min(outputs, axis=1).values
    alpha = np.quantile(confidence.detach().cpu().numpy(), confidence_q)
    indices = torch.where(confidence >= alpha)[0].detach().cpu()
    
    
    if len(indices) == 0:
        print("  Warning: No confident samples found. Skipping training step.")
        return

    # 3. Get hard pseudo-labels
    preds = torch.argmax(outputs, axis=1)
    
    # 4. Train student on confident pseudo-labeled data
    confident_x = torch.FloatTensor(unsup_x)[indices].detach().cpu()
    confident_y = preds[indices].detach().cpu()
    confident_x = confident_x.to(DEVICE)
    confident_y = confident_y.to(DEVICE)
    student = student.to(DEVICE)
    
    pseudo_dataset = TensorDataset(confident_x, confident_y)
    pseudo_loader = DataLoader(pseudo_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    optimizer = optim.Adam(student.parameters())
    criterion = nn.CrossEntropyLoss() 
    return train(student,  criterion, optimizer,  pseudo_loader, pseudo_loader,  verbose=False)

def gradual_self_train(model_class, teacher, unsup_x, unsup_y, target_test_loader, interval=2000, confidence_q=0.1, epochs=10):
    """Performs self-training on sequential chunks of unsupervised data."""
    num_steps = int(unsup_x.shape[0] / interval)
    student = model_class()
    for i in range(num_steps):
        print(f"  Gradual step {i+1}/{num_steps}")
        cur_xs = unsup_x[interval*i:interval*(i+1)]
        cur_ys = unsup_y[interval*i:interval*(i+1)]
        train_loader = DataLoader(TensorDataset(torch.FloatTensor(cur_xs), torch.LongTensor(cur_ys)), batch_size=BATCH_SIZE, shuffle=True)
        student = self_train_once(student, teacher, cur_xs, confidence_q, epochs)        
        teacher = student
        evaluate(train_loader, teacher, nn.CrossEntropyLoss() )
    return student

def gradual_self_train_2(model_class, teacher,  unsup_x, unsup_y , target_test_loader, interval=107, confidence_q=0.1, epochs=epoch):
    """Performs self-training on sequential chunks of unsupervised data."""
    num_steps = int(unsup_x.shape[0] / interval)
    student = model_class()
    for i in range(num_steps):
        print(f"  Gradual step {i+1}/{num_steps}")
        cur_xs = unsup_x[interval*i:interval*(i+1)]
        cur_ys = unsup_y[interval*i:interval*(i+1)]
        train_loader = DataLoader(TensorDataset(torch.FloatTensor(cur_xs), torch.LongTensor(cur_ys)), batch_size=BATCH_SIZE, shuffle=True)
        student = self_train_once(student, teacher, cur_xs, confidence_q, epochs)        
        teacher = student
    evaluate(train_loader, teacher, nn.CrossEntropyLoss() )

def train( source_model ,criterion, optimizer , src_train_loader , src_val_loader ,verbose=True ):
    global epoch
    if verbose :  
        global TEST_STEP_SIZE
    else :
        TEST_STEP_SIZE = -1 # this is so that the printing can be stopped or accuracy calculation in the middle 

    for epochs in range(epoch):
            total_loss = 0 
            correct = 0 
            total = 0 
            step = 0 
            print("Training Accuracy", end=" ")
            print(f'Epoch [{epochs+1}/{epoch}]')
            for data, target in src_train_loader:
                source_model.train() 
                data , target = data.to(DEVICE) , target.to(DEVICE)
                output = source_model(data)
                loss = criterion(output , target)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}',end=" ")
                
                # ---Source Validation ---# 
                if step==TEST_STEP_SIZE : 
                    source_model.eval()
                    step = 0 
                    with torch.no_grad():
                        for data, target in src_val_loader :
                            data, target = data.to(DEVICE) , target.to(DEVICE)
                            outputs = source_model(data)
                            _ , predicted_labels = torch.max(outputs.data, 1)
                            src_loss = criterion(outputs , target)
                            
                            total += target.size(0)
                            correct += (predicted_labels == target).sum().item()
                            total_loss += src_loss.item() * data.size(0)
                    src_avg_loss = total_loss / total 
                    src_accuracy = correct / total 
                                           
                    print(f'Loss : {src_avg_loss : .4f} ,Accuracy : {src_accuracy : .4f}')
                step +=1
    return source_model         

def evaluate(target_eval_loader , source_model , criterion ):
    # ---Target Validation ---# 
    total_loss = 0 
    total = 0
    correct = 0
    source_model.eval()
    with torch.no_grad():
        for data, target in target_eval_loader :
            data, target = data.to(DEVICE) , target.to(DEVICE)
            outputs = source_model(data)
            _ , predicted_labels = torch.max(outputs.data, 1)
            target_loss = criterion(outputs , target)
        
            total += target.size(0)
            correct += (predicted_labels == target).sum().item()
            total_loss += target_loss.item() * data.size(0)
    
    tgt_avg_loss = total_loss / total 
    tgt_accuracy = correct / total
    print(f'TEST : tgt_loss : {tgt_avg_loss : .4f}, tgt_accuracy : {tgt_accuracy : .4f} ')

def run_experiment( dataset_func , n_classes , save_file , model_class, interval , 
                   epochs, loss , soft , conf_q = 0.1 , num_runs = 5, num_repeats = None, find_upper_idx = False):
    global epoch
    (src_train_x , src_train_y , src_val_X , src_val_y , inter_x, inter_y ,  dir_inter_x , dir_inter_y , target_val_X, target_val_y , target_test_X, target_test_y) = dataset_func()
    print("-" * 50)
    print(f'Source training data shape: {src_train_x.shape}')
    print(f'Intermediate unsupervised data shape: {inter_x.shape}')
    print(f'Direct bootstrap (target) data shape: {dir_inter_x.shape}')
    print(f'Target test data shape: {target_test_X.shape}')
    print("-" * 50)

    print(f'Number of repeats: {num_repeats}\nNumber of runs: {num_runs}\nSoft training: {soft}\n'
          f'Confidence thresholding: {conf_q}\nInterval: {interval}\nEpochs: {epochs}\nLoss: {loss}')
    print("-" * 50)

    results = [] 
    for run in range(num_runs):
        print(f"Run {run + 1}/{num_runs}")
        
        print('\n ----------- Training source Model ----------')
        
        # --------------- 1.1 Data Source Loader ----------- #
        
        src_train_loader = DataLoader(TensorDataset(torch.FloatTensor(src_train_x), torch.LongTensor(src_train_y)), batch_size=BATCH_SIZE, shuffle=True)
        src_val_loader = DataLoader(TensorDataset(torch.FloatTensor(src_val_X), torch.LongTensor(src_val_y)), batch_size=BATCH_SIZE)
        target_eval_loader = DataLoader(TensorDataset(torch.FloatTensor(target_val_X), torch.LongTensor(target_val_y)), batch_size=BATCH_SIZE)
        target_test_loader = DataLoader(TensorDataset(torch.FloatTensor(target_test_X) , torch.LongTensor(target_test_y)),batch_size = BATCH_SIZE)
        # ---------------1.2 Creating Source Model -----------#
        source_model = model_class().to(DEVICE)
        optimizer = optim.Adam(source_model.parameters())
        criterion = nn.CrossEntropyLoss()
        
        # ---------------1.3 Training Source Model -----------#
        source_model = train(source_model , criterion , optimizer , src_train_loader , src_val_loader , target_eval_loader)
        print("Test Accuracy")
        # ----------------1.4 Testing With Target Dataset --------- # 
        evaluate(target_test_loader , source_model , criterion)
        
        
        # -----------------2 Training Gradually -------------#
        # --------------- 2.1 Creating Source Model -----------#
        teacher_gda = model_class().to(DEVICE)
        teacher_gda.load_state_dict(source_model.state_dict())
        student_gda =  gradual_self_train(
            model_class, teacher_gda, inter_x, inter_y, target_test_loader, interval, epochs=epochs, confidence_q=conf_q)
        evaluate(target_test_loader , student_gda , criterion)

def save_and_clear(model, save_file):
    torch.save(model.state_dict(), save_file)
    del model
    torch.cuda.empty_cache()
    gc.collect()

def load_model(model , file_name):
    model.load_state_dict(torch.load(file_name))
    return model

def create_mel(mel , datasource):
    

def run_experiment_2( dataset_func , n_classes , save_file , model_class, interval , 
                   epochs, loss , soft , conf_q = 0.1 , num_runs = 5, num_repeats = None, find_upper_idx = False):
    global epoch
    src_devices_mixup , target_devices_mixup, label_list , train_src_df, train_tgt_df, test_src_df, test_tgt_df  = dataset_func()

    print(f'Number of repeats: {num_repeats}\nNumber of runs: {num_runs}\nSoft training: {soft}\n'
          f'Confidence thresholding: {conf_q}\nInterval: {interval}\nEpochs: {epochs}\nLoss: {loss}')
    print("-" * 50)

    results = [] 
    for run in range(num_runs):
        print(f"Run {run + 1}/{num_runs}")
        
        print('\n ----------- Training source Model ----------')
        
        # --------------- 1.1 Data Source Loader ----------- #
        
        src_df = pd.DataFrame({ 'filename': src_devices_mixup, 'label': label_list })
        tgt_df = pd.DataFrame({  'filename': target_devices_mixup, 'label': label_list })

        # This all for paired
        target_sr = 32000
        target_duration = 10
        mel2 = AugmentMelSTFT2(n_mels=128, sr=32000, win_length=800, hopsize=320, n_fft=1024, freqm=48, timem=192, htk=False, fmin=0.0, fmax=None, norm=1, fmin_aug_range=10, fmax_aug_range=2000)
        
        # alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        alphas = [ ]
        mixed_spectrogram_x = [] 
        mixed_spectrogram_y = []
        for alpha in alphas:
            alpha_mel_mix = []
            label_for_each_domain = []
            for i in range(len(src_df)):
                source_path  = os.path.join( PATH , src_df.iloc[i]['filename'])
                dest_path  = os.path.join( PATH , tgt_df.iloc[i]['filename'])
                waveform1, sr1 = torchaudio.load(source_path)
                waveform2, sr2 = torchaudio.load(dest_path)
    
                if sr1 != target_sr:
                    resampler = T.Resample(orig_freq=sr1, new_freq=target_sr)
                    waveform1 = resampler(waveform1)
                if sr2 != target_sr:
                    resampler = T.Resample(orig_freq=sr2, new_freq=target_sr)
                    waveform2 = resampler(waveform2)
                # Squeeze to remove the channel dimension, making it (num_samples,)
                
                output = mel2(waveform1 , waveform2 , alpha)
                alpha_mel_mix.append(output)
                label_for_each_domain.append(label_keys[label_list[i]])
                
            label_for_each_domain = torch.tensor(label_for_each_domain)
            alpha_mel_mix = torch.stack( alpha_mel_mix , axis = 0 ) 
            mixed_spectrogram_x.append(alpha_mel_mix)
            mixed_spectrogram_y.append(label_for_each_domain)
        ################# the above code , please insert alphas 
        
        mixed_spectrogram_x = torch.load('saved_files/mixed_spectrogram_x.pt')
        mixed_spectrogram_y = torch.load('saved_files/mixed_spectrogram_y.pt')
        mixed_spectrogram_x = torch.cat(mixed_spectrogram_x, axis= 0 )
        mixed_spectrogram_y = torch.cat(mixed_spectrogram_y, axis = 0 )
        
      
        mel = AugmentMelSTFT(n_mels=128, sr=32000, win_length=800, hopsize=320, n_fft=1024, freqm=48,
                            timem=192,
                            htk=False, fmin=0.0, fmax=None, norm=1, fmin_aug_range=10,
                            fmax_aug_range=2000)
        # I dont have all the src and tgt that is why this new one
        
        ########### Train Source ########### 
        train_src_df = train_src_df['a']
        src_train_dataset = AudioDataset(x_col=train_src_df['filename'],y_col=train_src_df['scene_label'],  mel = mel, target_sr=target_sr, target_duration_sec=target_duration
        )
        src_train_loader= DataLoader( dataset=src_train_dataset, batch_size=4, shuffle=True)
        
        ######### Train Target ##############
        train_tgt_df = train_tgt_df['b']
        target_train_dataset = AudioDataset(x_col=train_tgt_df['filename'], y_col=train_tgt_df['scene_label'],  mel = mel, target_sr=target_sr, target_duration_sec=target_duration
        )
        target_train_loader= DataLoader( dataset=target_train_dataset, batch_size=4, shuffle=True)
        
        ######### Test Source ##############
        test_src_df = test_src_df['a']
        src_test_dataset = AudioDataset(x_col=test_src_df['filename'], y_col=test_src_df['scene_label'], mel = mel, target_sr=target_sr,
            target_duration_sec=target_duration
        )
        src_test_loader= DataLoader( dataset=src_test_dataset, batch_size=4, shuffle=True)
        
        ######### Test Target ##############
        test_tgt_df = test_tgt_df['b']
        target_test_dataset = AudioDataset(x_col=test_tgt_df['filename'], y_col=test_tgt_df['scene_label'], mel = mel, target_sr=target_sr,
            target_duration_sec=target_duration
        )
        target_test_loader= DataLoader( dataset=target_test_dataset, batch_size=4, shuffle=True)
        
        # ---------------1.2 Creating Source Model -----------#
        source_model = model_class() # Wrap the model for DataParallel
        source_model = source_model.to(DEVICE)    
        optimizer = optim.Adam(source_model.parameters(), lr = 1e-3)
        criterion = nn.CrossEntropyLoss()
        
        # ---------------1.3 Training Source Model -----------#
        # source_model = train(source_model , criterion , optimizer , src_train_loader , src_test_loader , target_test_loader)
        # print("Test Accuracy")
        # # ----------------1.4 Testing With Target Dataset --------- # 
        # evaluate(target_test_loader , source_model , criterion)
        
        
        # -----------------2 Training Gradually -------------#
        # --------------- 2.1 Creating Source Model -----------#
        
        teacher_gda = model_class().to(DEVICE)
        # teacher_gda.load_state_dict(source_model.state_dict(), )
        source_file_name = 'source_model.pth'
        save_and_clear(source_model , os.path.join(SAVE_DIR , source_file_name))
        student_gda =  gradual_self_train_2(
            model_class, teacher_gda, mixed_spectrogram_x , mixed_spectrogram_y, target_test_loader, interval=749, epochs=epochs, confidence_q=conf_q)
        evaluate(target_test_loader , student_gda , criterion)
                    
run_experiment_2(Passt_with_mix, epoch , 'saved_files/mnist.pkl' , PaSST_wrapper, 2000 , 10 , 'ce' , True , 0.3 , num_runs , None , False )
        
    