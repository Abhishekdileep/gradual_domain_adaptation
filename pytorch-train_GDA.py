# main_pytorch.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pickle
import ipdb  # Debugger
import copy
from dataset_mix_up import rotated_mnist_mix_up_data_func # Mock dataset function
import librosa
from hear21passt.base import load_model, get_scene_embeddings, get_timestamp_embeddings , get_basic_model_2
from hear21passt.passt import get_model_passt
from hear21passt.wrapper import PasstBasicWrapper2
from Augment_mel_stft_mixup import AugmentMelSTFT2
import numpy as np 

# =============================================================================
# 1. Model Definition (PyTorch)
# =============================================================================

class PaSSTFeatureExtractor(torch.nn.Module):
    def __init__(self, mel ,  device=None):
        super(PaSSTFeatureExtractor, self).__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # self.model = get_basic_model(mode="embed_only") 
        # self.model.to(self.device)
        self.mel = mel 
        self.net = get_model_passt(arch="passt_s_swa_p16_128_ap476")
        self.model = PasstBasicWrapper2(mel=self.mel, net=self.net)
        self.model.to(self.device)

    def forward(self, audio_waveform, sample_rate=32000):
        if audio_waveform.dim() == 1:
            audio_waveform = audio_waveform.unsqueeze(0)  

        audio_waveform = audio_waveform.to(self.device)
        
        # Allow gradients to flow through PaSST for domain adaptation
        features = self.model(audio_waveform)
             
        return features


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

# =============================================================================
# 2. Utility Functions (PyTorch)
# =============================================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

def rand_seed(seed):
    """Sets random seeds for reproducibility."""
    # ipdb.set_trace()
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_loss(loss_name='ce'):
    """Returns a PyTorch loss function."""
    if loss_name == 'ce':
        # CrossEntropyLoss in PyTorch can handle both hard and soft labels.
        return nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Loss '{loss_name}' not supported.")

# =============================================================================
# 3. Core Training & Evaluation Loops (PyTorch)
# =============================================================================

def train_model(model, dataloader, criterion, optimizer, epochs, verbose=True):
    """
    Generic training loop for a PyTorch model.
    Replaces Keras's `.fit()` method.
    """
    model.train() # Set model to training mode
    for epoch in range(epochs):
        for data, targets in dataloader:
            data, targets = data.to(DEVICE), targets.to(DEVICE)
            
            # Forward pass
            outputs = model(data)
            loss = criterion(outputs, targets)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        if verbose:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

def evaluate_model(model, dataloader):
    """
    Generic evaluation loop for a PyTorch model.
    Replaces Keras's `.evaluate()` method.
    Returns loss and accuracy.
    """
    model.eval() # Set model to evaluation mode
    total_loss = 0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for data, targets in dataloader:
            data, targets = data.to(DEVICE), targets.to(DEVICE)
            outputs = model(data)
            
            # Check if targets are one-hot encoded (soft labels)
            if targets.ndim > 1 and targets.shape[1] > 1:
                # For soft labels, we can't directly compute accuracy in the same way.
                # We'll compare the argmax of prediction and target.
                _, predicted_labels = torch.max(outputs.data, 1)
                _, true_labels = torch.max(targets.data, 1)
                loss = criterion(outputs, targets)
            else:
                # Hard labels
                _, predicted_labels = torch.max(outputs.data, 1)
                true_labels = targets
                loss = criterion(outputs, true_labels)

            total += true_labels.size(0)
            correct += (predicted_labels == true_labels).sum().item()
            total_loss += loss.item() * data.size(0)
            
    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy

def predict(model, dataloader):
    """
    Generates predictions (logits) for a given dataset.
    Replaces Keras's `.predict()` method.
    """
    model.eval()
    all_outputs = []
    with torch.no_grad():
        for data, _ in dataloader:
            data = data.to(DEVICE)
            outputs = model(data)
            all_outputs.append(outputs.cpu())
    return torch.cat(all_outputs, dim=0)


# =============================================================================
# 4. Self-Training Algorithms (PyTorch)
# =============================================================================

def soft_self_train_once(student, teacher, unsup_x, epochs=20):
    """Trains student on soft pseudo-labels from teacher."""
    # ipdb.set_trace()
    print("  Performing one soft self-training step...")
    # 1. Get teacher's predictions (soft labels)
    unsup_dataset = TensorDataset(torch.FloatTensor(unsup_x), torch.zeros(len(unsup_x))) # dummy labels
    unsup_loader = DataLoader(unsup_dataset, batch_size=128, shuffle=False)
    
    teacher_probs = torch.softmax(predict(teacher, unsup_loader), dim=1)

    # 2. Train student on these soft labels
    pseudo_dataset = TensorDataset(torch.FloatTensor(unsup_x), teacher_probs)
    pseudo_loader = DataLoader(pseudo_dataset, batch_size=128, shuffle=True)
    
    optimizer = optim.Adam(student.parameters())
    criterion = get_loss('ce')
    
    train_model(student, pseudo_loader, criterion, optimizer, epochs, verbose=False)

def self_train_once(student, teacher, unsup_x, confidence_q=0.1, epochs=20):
    """Trains student on hard pseudo-labels from teacher, with confidence filtering."""
    # ipdb.set_trace()
    print("  Performing one hard self-training step...")
    # 1. Get teacher's predictions (logits)
    unsup_dataset = TensorDataset(torch.FloatTensor(unsup_x), torch.zeros(len(unsup_x))) # dummy labels
    unsup_loader = DataLoader(unsup_dataset, batch_size=128, shuffle=False)
    
    logits = predict(teacher, unsup_loader)
    
    # 2. Confidence-based filtering
    confidence = torch.max(logits, axis=1).values - torch.min(logits, axis=1).values
    alpha = np.quantile(confidence.numpy(), confidence_q)
    indices = torch.where(confidence >= alpha)[0]
    
    if len(indices) == 0:
        print("  Warning: No confident samples found. Skipping training step.")
        return

    # 3. Get hard pseudo-labels
    preds = torch.argmax(logits, axis=1)
    
    # 4. Train student on confident pseudo-labeled data
    confident_x = torch.FloatTensor(unsup_x)[indices]
    confident_y = preds[indices]
    
    pseudo_dataset = TensorDataset(confident_x, confident_y)
    pseudo_loader = DataLoader(pseudo_dataset, batch_size=128, shuffle=True)
    
    optimizer = optim.Adam(student.parameters())
    criterion = get_loss('ce')
    
    train_model(student, pseudo_loader, criterion, optimizer, epochs, verbose=False)

def self_train(student_func, teacher, unsup_x, confidence_q=0.1, epochs=20, repeats=1,
               target_x=None, target_y=None, soft=False):
    """Repeatedly performs self-training."""
    # ipdb.set_trace()
    accuracies = []
    for i in range(repeats):
        print(f"  Self-train repeat {i+1}/{repeats}")
        student = student_func(teacher)
        if soft:
            soft_self_train_once(student, teacher, unsup_x, epochs)
        else:
            self_train_once(student, teacher, unsup_x, confidence_q, epochs)
            
        if target_x is not None and target_y is not None:
            target_dataset = TensorDataset(torch.FloatTensor(target_x), torch.LongTensor(target_y))
            target_loader = DataLoader(target_dataset, batch_size=128)
            _, accuracy = evaluate_model(student, target_loader)
            print(f"  Accuracy after repeat {i+1}: {accuracy:.4f}")
            accuracies.append(accuracy)
            
        teacher = student # Student becomes the new teacher
    return accuracies, student

def gradual_self_train(student_func, teacher, unsup_x, debug_y, interval, confidence_q=0.1, epochs=20, soft=False, upper_idx=None):
    """Performs self-training on sequential chunks of data."""
    # ipdb.set_trace()
    if upper_idx is None:
        upper_idx = int(unsup_x.shape[0] / interval)
        
    accuracies = []
    for i in range(upper_idx):
        print(f"Gradual step {i+1}/{upper_idx}")
        student = student_func(teacher)
        start, end = interval * i, interval * (i + 1)
        cur_xs = unsup_x[start:end]
        cur_ys = debug_y[start:end]
        
        if soft:
            soft_self_train_once(student, teacher, cur_xs, epochs)
        else:
            self_train_once(student, teacher, cur_xs, confidence_q, epochs)
        
        # Evaluate on the current chunk of data
        cur_dataset = TensorDataset(torch.FloatTensor(cur_xs), torch.LongTensor(cur_ys))
        cur_loader = DataLoader(cur_dataset, batch_size=128)
        _, accuracy = evaluate_model(student, cur_loader)
        accuracies.append(accuracy)
        print(f"  Accuracy on current chunk: {accuracy:.4f}")
        
        teacher = student # Student becomes the new teacher
    return accuracies, student

# =============================================================================
# 5. Experiment Orchestration (PyTorch)
# =============================================================================

def run_experiment(
    dataset_func, n_classes, save_file, model_func=SimpleConvNet, interval=2000, epochs=10, loss='ce', 
    soft=False, conf_q=0.1, num_runs=5, num_repeats=None, find_upper_idx=False):
    
    (src_tr_x, src_tr_y, src_val_x, src_val_y,
     inter_x, inter_y, dir_inter_x, dir_inter_y,
     trg_val_x, trg_val_y, trg_test_x, trg_test_y) = dataset_func()

    # Log dataset shapes
    print("-" * 50)
    print(f'Source training data shape: {src_tr_x.shape}')
    print(f'Intermediate unsupervised data shape: {inter_x.shape}')
    print(f'Direct bootstrap (target) data shape: {dir_inter_x.shape}')
    print(f'Target test data shape: {trg_test_x.shape}')
    print("-" * 50)

    upper_idx = None
    if find_upper_idx:
        # ipdb.set_trace()
        upper_idx = min(src_tr_x.shape[0], dir_inter_x.shape[0]) // interval

    if num_repeats is None:
        num_repeats = int(inter_x.shape[0] / interval)

    print(f'Number of repeats: {num_repeats}\nNumber of runs: {num_runs}\nSoft training: {soft}\n'
          f'Confidence thresholding: {conf_q}\nInterval: {interval}\nEpochs: {epochs}\nLoss: {loss}')
    print("-" * 50)

    def new_model():
        model = model_func(n_classes).to(DEVICE)
        return model

    def student_func(teacher):
        # Create a new student model with the same weights as the teacher
        student = new_model()
        student.load_state_dict(teacher.state_dict())
        return student

    def run(seed):
        rand_seed(seed)
        
        # Create DataLoaders
        src_tr_loader = DataLoader(TensorDataset(torch.FloatTensor(src_tr_x), torch.LongTensor(src_tr_y)), batch_size=128, shuffle=True)
        src_val_loader = DataLoader(TensorDataset(torch.FloatTensor(src_val_x), torch.LongTensor(src_val_y)), batch_size=128)
        trg_eval_loader = DataLoader(TensorDataset(torch.FloatTensor(trg_test_x), torch.LongTensor(trg_test_y)), batch_size=128)

        # 1. Train source model
        source_model = new_model()
        optimizer = optim.Adam(source_model.parameters())
        criterion = get_loss(loss)
        print("\n--- Training Source Model ---")
        train_model(source_model, src_tr_loader, criterion, optimizer, epochs, verbose=False)
        
        _, src_acc = evaluate_model(source_model, src_val_loader)
        _, target_acc = evaluate_model(source_model, trg_eval_loader)
        print(f"Source Model Accuracy on Source Val: {src_acc:.4f}")
        print(f"Source Model Accuracy on Target Test: {target_acc:.4f}")

        # 2. Gradual self-training (GDA)
        print("\n--- Gradual Self-Training (GDA) ---")
        teacher_gda = new_model()
        teacher_gda.load_state_dict(source_model.state_dict())
        gradual_accuracies, student_gda = gradual_self_train(
            student_func, teacher_gda, inter_x, inter_y, interval, 
            epochs=epochs, soft=soft, confidence_q=conf_q, upper_idx=upper_idx
        )
        _, acc_gda = evaluate_model(student_gda, trg_eval_loader)
        gradual_accuracies.append(acc_gda)
        print(f"Final GDA accuracy on target: {acc_gda:.4f}")

        # 3. Direct bootstrap to target
        print("\n--- Direct Bootstrap to Target ---")
        teacher_direct = new_model()
        teacher_direct.load_state_dict(source_model.state_dict())
        target_accuracies, _ = self_train(
            student_func, teacher_direct, dir_inter_x, epochs=epochs, 
            target_x=trg_test_x, target_y=trg_test_y, repeats=num_repeats, 
            soft=soft, confidence_q=conf_q
        )
        print(f"Final Direct Bootstrap accuracy on target: {target_accuracies[-1]:.4f}")

        # 4. Direct bootstrap to all unsupervised data
        print("\n--- Direct Bootstrap to All Unsupervised Data ---")
        teacher_all = new_model()
        teacher_all.load_state_dict(source_model.state_dict())
        all_accuracies, _ = self_train(
            student_func, teacher_all, inter_x, epochs=epochs, 
            target_x=trg_test_x, target_y=trg_test_y, repeats=num_repeats, 
            soft=soft, confidence_q=conf_q
        )
        print(f"Final All-Unsup Bootstrap accuracy on target: {all_accuracies[-1]:.4f}")

        return src_acc, target_acc, gradual_accuracies, target_accuracies, all_accuracies

    results = []
    for i in range(num_runs):
        print(f'\n================ RUN {i + 1}/{num_runs} ================')
        results.append(run(i))
        
    print(f'\nSaving results to {save_file}')
    with open(save_file, "wb") as f:
        pickle.dump(results, f)

def experiment_results(save_name):
    """Loads and prints formatted results from a pickle file."""
    # ipdb.set_trace()
    try:
        with open(save_name, "rb") as f:
            results = pickle.load(f)
    except FileNotFoundError:
        print(f"Error: Results file not found at '{save_name}'")
        return

    src_accs, target_accs = [], []
    final_graduals, final_targets, final_alls = [], [], []
    best_targets, best_alls = [], []

    for res in results:
        src_acc, target_acc, gradual_accuracies, target_accuracies, all_accuracies = res
        src_accs.append(100 * src_acc)
        target_accs.append(100 * target_acc)
        final_graduals.append(100 * gradual_accuracies[-1])
        final_targets.append(100 * target_accuracies[-1])
        final_alls.append(100 * all_accuracies[-1])
        best_targets.append(100 * np.max(target_accuracies))
        best_alls.append(100 * np.max(all_accuracies))

    num_runs = len(src_accs)
    mult = 1.96  # For 95% confidence intervals

    def print_stat(name, data):
        mean = np.mean(data)
        std_err = mult * np.std(data) / np.sqrt(num_runs)
        print(f"{name:<40}: {mean:.2f} ± {std_err:.2f}")

    print("\n" + "="*25 + " FINAL RESULTS " + "="*25)
    print_stat("Non-adaptive accuracy on source (%)", src_accs)
    print_stat("Non-adaptive accuracy on target (%)", target_accs)
    print_stat("Gradual self-train accuracy (%)", final_graduals)
    print_stat("Target self-train accuracy (final) (%)", final_targets)
    print_stat("All self-train accuracy (final) (%)", final_alls)
    print_stat("Target self-train accuracy (best) (%)", best_targets)
    print_stat("All self-train accuracy (best) (%)", best_alls)
    print("="*67)

# =============================================================================
# 6. Main Execution Block
# =============================================================================

def mixup_mnist_60_conv_experiment_pytorch():
    """Defines and runs a specific experiment configuration."""
    # ipdb.set_trace()
    run_experiment(
        dataset_func=rotated_mnist_mix_up_data_func, 
        n_classes=10, 
        save_file='saved_files/mixup_mnist_60_conv_pytorch.pkl',
        model_func=SimpleConvNet, 
        interval=2000, # Reduced for faster execution
        epochs=5,      # Reduced for faster execution
        loss='ce',
        soft=False, 
        conf_q=0.1, 
        num_runs=2,    # Reduced for faster execution
        num_repeats=3, # Reduced for faster execution
        find_upper_idx=False
    )

if __name__ == "__main__":
    # ipdb.set_trace()
    
    # Create directory for saved files if it doesn't exist
    import os
    if not os.path.exists('saved_files'):
        os.makedirs('saved_files')

    print("--- Running Mixup MNIST PyTorch Experiment ---")
    mixup_mnist_60_conv_experiment_pytorch()
    
    print("\n--- Analyzing Mixup MNIST PyTorch Experiment Results ---")
    experiment_results('saved_files/mixup_mnist_60_conv_pytorch.pkl')

# dataset_mix_up.py

import numpy as np
import torch
from torchvision.datasets import MNIST
from torchvision.transforms import functional as F
from scipy.ndimage import rotate

def rotated_mnist_mix_up_data_func(source_angle=0, target_angle=60, num_intermediate_domains=5):
    """
    Creates a mock dataset for domain adaptation experiments using rotated MNIST.
    This function replaces the missing dataset file from the original code.

    - Source Domain: MNIST digits rotated by `source_angle`.
    - Target Domain: MNIST digits rotated by `target_angle`.
    - Intermediate Domain: A sequence of domains with rotations smoothly
      interpolating between the source and target angles.

    Returns:
        A tuple of numpy arrays matching the structure of the original experiment.
    """
    print("--- Generating Rotated MNIST Dataset ---")
    
    # 1. Load MNIST data
    mnist_train = MNIST(root='./data', train=True, download=True)
    mnist_test = MNIST(root='./data', train=False, download=True)
    
    x_train, y_train = mnist_train.data.numpy(), mnist_train.targets.numpy()
    x_test, y_test = mnist_test.data.numpy(), mnist_test.targets.numpy()

    # Normalize and add channel dimension placeholder
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.

    def apply_rotation(images, angle):
        if angle == 0:
            return images
        return rotate(images, angle, axes=(1, 2), reshape=False)

    # 2. Create Source Domain Data
    # Using first 10000 samples for the source domain
    src_x = apply_rotation(x_train[:10000], source_angle)
    src_y = y_train[:10000]
    src_tr_x, src_val_x = src_x[:8000], src_x[8000:]
    src_tr_y, src_val_y = src_y[:8000], src_y[8000:]

    # 3. Create Target Domain Data
    # Using test set for target evaluation
    trg_x = apply_rotation(x_test, target_angle)
    trg_y = y_test
    trg_val_x, trg_test_x = trg_x[:5000], trg_x[5000:]
    trg_val_y, trg_test_y = trg_y[:5000], trg_y[5000:]
    
    # `dir_inter_x` is the unlabeled data from the target domain used for direct bootstrapping
    # Let's use the first 6000 samples from the training set, rotated to the target angle
    dir_inter_x = apply_rotation(x_train[20000:26000], target_angle)
    dir_inter_y = y_train[20000:26000] # Labels are for debugging/evaluation only

    # 4. Create Intermediate Domain Data (`inter_x`)
    # This data smoothly transitions from source to target
    angles = np.linspace(source_angle, target_angle, num_intermediate_domains + 2)[1:-1]
    
    inter_x_list = []
    inter_y_list = []
    
    # Use a chunk of training data for the intermediate domains
    num_samples_per_chunk = 3000
    start_idx = 30000
    
    for i, angle in enumerate(angles):
        print(f"Generating intermediate domain {i+1}/{len(angles)} with angle {angle:.1f}")
        chunk_start = start_idx + i * num_samples_per_chunk
        chunk_end = chunk_start + num_samples_per_chunk
        
        unrotated_chunk = x_train[chunk_start:chunk_end]
        rotated_chunk = apply_rotation(unrotated_chunk, angle)
        
        inter_x_list.append(rotated_chunk)
        inter_y_list.append(y_train[chunk_start:chunk_end])
        
    inter_x = np.concatenate(inter_x_list, axis=0)
    inter_y = np.concatenate(inter_y_list, axis=0)

    # Add channel dimension for Conv2D layers
    # PyTorch expects (N, C, H, W)
    src_tr_x = np.expand_dims(src_tr_x, 1)
    src_val_x = np.expand_dims(src_val_x, 1)
    inter_x = np.expand_dims(inter_x, 1)
    dir_inter_x = np.expand_dims(dir_inter_x, 1)
    trg_val_x = np.expand_dims(trg_val_x, 1)
    trg_test_x = np.expand_dims(trg_test_x, 1)

    print("--- Dataset Generation Complete ---")
    
    return (
        src_tr_x, src_tr_y, src_val_x, src_val_y,
        inter_x, inter_y, dir_inter_x, dir_inter_y,
        trg_val_x, trg_val_y, trg_test_x, trg_test_y
    )

