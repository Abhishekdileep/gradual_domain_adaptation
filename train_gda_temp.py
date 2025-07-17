import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import numpy as np
import pickle
from torchvision.datasets import MNIST
from scipy.ndimage import rotate
import copy
from dataset_mix_up import rotated_mnist_60_data_func

# =============================================================================
# 1. Parameters
# =============================================================================

BATCH_SIZE = 64
NUM_RUNS = 5
EPOCHS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
NUM_CLASSES = 10
INTERVAL = 2000 # Number of samples per gradual step

# =============================================================================
# 2. Model Definition (PyTorch)
# =============================================================================

class SimpleConvNet(nn.Module):
    """
    A simple Convolutional Neural Network model, equivalent to the Keras
    model `simple_softmax_conv_model` from the reference implementation.
    """
    def __init__(self, num_classes=10):
        super(SimpleConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        if x.dim() == 2:
            x = x.view(-1, 28, 28)
        if x.dim() == 3:
            x = x.unsqueeze(1)
        out = self.pool1(self.relu1(self.conv1(x)))
        out = self.pool2(self.relu2(self.conv2(out)))
        out = out.view(out.size(0), -1)
        out = self.relu3(self.fc1(out))
        out = self.fc2(out)
        return out

# =============================================================================
# 3. Dataset Function
# =============================================================================

def rotated_mnist_60_data_func1():
    """
    Generates the Rotated MNIST dataset for the domain adaptation experiment.
    - Source Domain: MNIST digits with 0-degree rotation.
    - Target Domain: MNIST digits with 60-degree rotation.
    - Intermediate Domain: A sequence of domains with rotations smoothly
      interpolating between the source and target angles.
    """
    print("--- Generating Rotated MNIST Dataset ---")
    mnist_train = MNIST(root='./data', train=True, download=True)
    mnist_test = MNIST(root='./data', train=False, download=True)

    x_train_all, y_train_all = mnist_train.data.numpy(), mnist_train.targets.numpy()
    x_test_all, y_test_all = mnist_test.data.numpy(), mnist_test.targets.numpy()

    x_all = np.concatenate([x_train_all, x_test_all]).astype('float32') / 255.
    y_all = np.concatenate([y_train_all, y_test_all])
    
    # Shuffle the dataset
    np.random.seed(42)
    shuffled_indices = np.random.permutation(len(x_all))
    x_all, y_all = x_all[shuffled_indices], y_all[shuffled_indices]

    # Define dataset sizes
    src_train_size = 10000
    src_val_size = 2000
    inter_size = 40000
    dir_inter_size = 10000
    trg_val_size = 2000
    trg_test_size = 6000

    # Split data
    src_tr_x, src_tr_y = x_all[:src_train_size], y_all[:src_train_size]
    src_val_x, src_val_y = x_all[src_train_size:src_train_size+src_val_size], y_all[src_train_size:src_train_size+src_val_size]
    inter_x, inter_y = x_all[22000:62000], y_all[22000:62000]
    dir_inter_x, dir_inter_y = x_all[62000:70000], y_all[62000:70000] # Use last part for direct
    trg_val_x, trg_val_y = x_all[12000:14000], y_all[12000:14000] # Use a slice for target val
    trg_test_x, trg_test_y = x_test_all[:trg_test_size], y_test_all[:trg_test_size]

    # Apply rotations
    print("Applying rotations...")
    # Source data has 0-degree rotation (no change)
    # Intermediate data has gradual rotation from 0 to 60 degrees
    for i in range(len(inter_x)):
        angle = (i / len(inter_x)) * 60.0
        inter_x[i] = rotate(inter_x[i], angle, reshape=False, mode='nearest')
    
    # Direct intermediate and target data has 60-degree rotation
    for i in range(len(dir_inter_x)):
        dir_inter_x[i] = rotate(dir_inter_x[i], 60, reshape=False, mode='nearest')
    for i in range(len(trg_val_x)):
        trg_val_x[i] = rotate(trg_val_x[i], 60, reshape=False, mode='nearest')
    for i in range(len(trg_test_x)):
        trg_test_x[i] = rotate(trg_test_x[i], 60, reshape=False, mode='nearest')

    print("--- Dataset Generation Complete ---")
    return (src_tr_x, src_tr_y, src_val_x, src_val_y, inter_x, inter_y, 
            dir_inter_x, dir_inter_y, trg_val_x, trg_val_y, trg_test_x, trg_test_y)

def temp():
    (src_train_x , src_train_y , src_val_X , src_val_y , inter_x, inter_y ,  dir_inter_x , dir_inter_y , target_val_X, target_val_y , target_test_X, target_test_y) = rotated_mnist_60_data_func(20000, 22000, 48000, 50000)
    src_train_x = src_train_x.squeeze(3)
    src_val_X = src_val_X.squeeze(3)
    inter_x = inter_x.squeeze(3)
    dir_inter_x = dir_inter_x.squeeze(3)    
    target_val_X = target_val_X.squeeze(3)    
    target_test_X = target_test_X.squeeze(3)    
    return (src_train_x , src_train_y , src_val_X , src_val_y , inter_x, inter_y ,  dir_inter_x , dir_inter_y , target_val_X, target_val_y , target_test_X, target_test_y)


# =============================================================================
# 4. Utility Functions (PyTorch)
# =============================================================================

def get_loss(loss_name='ce'):
    """Returns a loss function."""
    if loss_name == 'ce':
        return nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unknown loss: {loss_name}")

def predict(model, loader):
    """Generates predictions (logits) for a given dataset."""
    model.eval()
    all_logits = []
    with torch.no_grad():
        for data, _ in loader:
            data = data.to(DEVICE)
            logits = model(data)
            all_logits.append(logits.cpu())
    return torch.cat(all_logits, dim=0)

def evaluate_model(model, loader, criterion=None):
    """Evaluates the model's performance."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)
            if criterion:
                total_loss += criterion(output, target).item() * data.size(0)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    avg_loss = total_loss / total if criterion else 0
    accuracy = correct / total
    return avg_loss, accuracy

def train_model(model, loader, criterion, optimizer, epochs, verbose=True):
    """A standard model training loop."""
    model.train()
    for epoch in range(epochs):
        for data, target in loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        if verbose and (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

# =============================================================================
# 5. Self-Training Functions (PyTorch)
# =============================================================================

def self_train_once(student, teacher, unsup_x, confidence_q=0.1, epochs=10):
    """Trains student on hard pseudo-labels from teacher, with confidence filtering."""
    print("  Performing one hard self-training step...")
    teacher.eval()
    unsup_dataset = TensorDataset(torch.from_numpy(unsup_x).float(), torch.zeros(len(unsup_x)))
    unsup_loader = DataLoader(unsup_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    logits = predict(teacher, unsup_loader)
    
    confidence = torch.max(logits, axis=1).values - torch.min(logits, axis=1).values
    alpha = np.quantile(confidence.numpy(), confidence_q)
    indices = torch.where(confidence >= alpha)[0]

    if len(indices) == 0:
        print("  Warning: No confident samples found. Skipping training step.")
        return

    preds = torch.argmax(logits, axis=1)
    
    confident_x = torch.from_numpy(unsup_x).float()[indices]
    confident_y = preds[indices]
    
    pseudo_dataset = TensorDataset(confident_x, confident_y)
    pseudo_loader = DataLoader(pseudo_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    optimizer = optim.Adam(student.parameters())
    criterion = get_loss('ce')
    
    train_model(student, pseudo_loader, criterion, optimizer, epochs, verbose=False)

def soft_self_train_once(student, teacher, unsup_x, epochs=10):
    """Trains student on soft pseudo-labels (probabilities) from teacher."""
    print("  Performing one soft self-training step...")
    teacher.eval()
    unsup_dataset = TensorDataset(torch.from_numpy(unsup_x).float(), torch.zeros(len(unsup_x)))
    unsup_loader = DataLoader(unsup_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    logits = predict(teacher, unsup_loader)
    probs = torch.softmax(logits, dim=1)
    
    pseudo_dataset = TensorDataset(torch.from_numpy(unsup_x).float(), probs)
    pseudo_loader = DataLoader(pseudo_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    optimizer = optim.Adam(student.parameters())
    criterion = get_loss('ce') # CrossEntropyLoss handles soft labels in PyTorch
    
    train_model(student, pseudo_loader, criterion, optimizer, epochs, verbose=False)

def self_train(student_func, teacher, unsup_x, confidence_q=0.1, epochs=10, repeats=1,
               target_x=None, target_y=None, soft=False):
    """Repeatedly performs self-training."""
    accuracies = []
    student = student_func(teacher)
    for i in range(repeats):
        print(f"  Self-train repeat {i+1}/{repeats}")
        if soft:
            soft_self_train_once(student, teacher, unsup_x, epochs)
        else:
            self_train_once(student, teacher, unsup_x, confidence_q, epochs)
            
        if target_x is not None and target_y is not None:
            target_dataset = TensorDataset(torch.from_numpy(target_x).float(), torch.from_numpy(target_y).long())
            target_loader = DataLoader(target_dataset, batch_size=BATCH_SIZE)
            _, accuracy = evaluate_model(student, target_loader)
            print(f"  Accuracy after repeat {i+1}: {accuracy:.4f}")
            accuracies.append(accuracy)
            
        teacher = student
    return accuracies, student

def gradual_self_train(student_func, teacher, unsup_x, interval, confidence_q=0.1, epochs=10, soft=False):
    """Performs self-training on sequential chunks of unsupervised data."""
    num_steps = int(unsup_x.shape[0] / interval)
    student = student_func(teacher)

    for i in range(num_steps):
        print(f"  Gradual step {i+1}/{num_steps}")
        cur_xs = unsup_x[interval*i:interval*(i+1)]
        
        if soft:
            soft_self_train_once(student, teacher, cur_xs, epochs)
        else:
            self_train_once(student, teacher, cur_xs, confidence_q, epochs)
        
        teacher = student
        
    return student

# =============================================================================
# 6. Experiment Functions
# =============================================================================

def run_experiment(dataset_func, n_classes, save_file, model_class, interval,
                   epochs, loss, soft, conf_q=0.1, num_runs=5):
    
    (src_tr_x, src_tr_y, src_val_x, src_val_y, inter_x, inter_y,
     dir_inter_x, dir_inter_y, trg_val_x, trg_val_y, trg_test_x, trg_test_y) = dataset_func()

    num_repeats = int(inter_x.shape[0] / interval)

    print("-" * 50)
    print(f'Number of repeats: {num_repeats}\nNumber of runs: {num_runs}\nSoft training: {soft}\n'
          f'Confidence thresholding: {conf_q}\nInterval: {interval}\nEpochs: {epochs}\nLoss: {loss}')
    print("-" * 50)

    def new_model():
        return model_class(n_classes).to(DEVICE)

    def student_func(teacher):
        # Create a new student model with the same weights as the teacher
        student = new_model()
        student.load_state_dict(teacher.state_dict())
        return student

    results = []
    for i in range(num_runs):
        print(f"\n{'='*20} Run {i + 1}/{num_runs} {'='*20}")
        np.random.seed(i)
        torch.manual_seed(i)

        # --- DataLoaders ---
        src_train_loader = DataLoader(TensorDataset(torch.from_numpy(src_tr_x).float(), torch.from_numpy(src_tr_y).long()), batch_size=BATCH_SIZE, shuffle=True)
        src_val_loader = DataLoader(TensorDataset(torch.from_numpy(src_val_x).float(), torch.from_numpy(src_val_y).long()), batch_size=BATCH_SIZE)
        trg_eval_loader = DataLoader(TensorDataset(torch.from_numpy(trg_val_x).float(), torch.from_numpy(trg_val_y).long()), batch_size=BATCH_SIZE)
        
        # --- 1. Train source model ---
        print("\n--- Training source model ---")
        source_model = new_model()
        optimizer = optim.Adam(source_model.parameters())
        criterion = get_loss(loss)
        train_model(source_model, src_train_loader, criterion, optimizer, epochs, verbose=True)
        
        _, src_acc = evaluate_model(source_model, src_val_loader)
        _, target_acc = evaluate_model(source_model, trg_eval_loader)
        print(f"Source-only accuracy on source val: {src_acc:.4f}")
        print(f"Source-only accuracy on target val: {target_acc:.4f}")

        # --- 2. Gradual self-training (GDA) ---
        print("\n\n--- Gradual self-training ---")
        teacher_gda = new_model()
        teacher_gda.load_state_dict(source_model.state_dict())
        student_gda = gradual_self_train(
            student_func, teacher_gda, inter_x, interval, epochs=epochs, soft=soft, confidence_q=conf_q)
        _, final_gradual_acc = evaluate_model(student_gda, trg_eval_loader)
        print(f"Final Gradual Self-Train Accuracy: {final_gradual_acc:.4f}")
        # Note: The original code returns accuracies at each step. Here we only get the final one.
        gradual_accuracies = [final_gradual_acc] 

        # --- 3. Direct bootstrap to target ---
        print("\n\n--- Direct bootstrap to target ---")
        teacher_direct_target = new_model()
        teacher_direct_target.load_state_dict(source_model.state_dict())
        target_accuracies, _ = self_train(
            student_func, teacher_direct_target, dir_inter_x, epochs=epochs, target_x=trg_val_x,
            target_y=trg_val_y, repeats=num_repeats, soft=soft, confidence_q=conf_q)

        # --- 4. Direct bootstrap to all unsupervised data ---
        print("\n\n--- Direct bootstrap to all unsupervised data ---")
        teacher_direct_all = new_model()
        teacher_direct_all.load_state_dict(source_model.state_dict())
        all_accuracies, _ = self_train(
            student_func, teacher_direct_all, inter_x, epochs=epochs, target_x=trg_val_x,
            target_y=trg_val_y, repeats=num_repeats, soft=soft, confidence_q=conf_q)

        results.append((src_acc, target_acc, gradual_accuracies, target_accuracies, all_accuracies))

    print(f'\nSaving results to {save_file}')
    with open(save_file, "wb") as f:
        pickle.dump(results, f)

def experiment_results(save_name):
    """Prints the mean and confidence intervals for the experiment results."""
    with open(save_name, "rb") as f:
        results = pickle.load(f)
        
    src_accs, target_accs = [], []
    final_graduals, final_targets, final_alls = [], [], []
    best_targets, best_alls = [], []
    
    for src_acc, target_acc, gradual_accuracies, target_accuracies, all_accuracies in results:
        src_accs.append(100 * src_acc)
        target_accs.append(100 * target_acc)
        if gradual_accuracies:
            final_graduals.append(100 * gradual_accuracies[-1])
        if target_accuracies:
            final_targets.append(100 * target_accuracies[-1])
            best_targets.append(100 * np.max(target_accuracies))
        if all_accuracies:
            final_alls.append(100 * all_accuracies[-1])
            best_alls.append(100 * np.max(all_accuracies))

    num_runs = len(src_accs)
    mult = 1.96  # For 95% confidence intervals

    def print_stat(name, data):
        if not data:
            print(f"{name}: N/A")
            return
        mean = np.mean(data)
        std_err = np.std(data) / np.sqrt(num_runs)
        print(f"{name} (%): {mean:.2f} +/- {mult * std_err:.2f}")

    print("\n--- Experiment Results ---")
    print_stat("Non-adaptive accuracy on source", src_accs)
    print_stat("Non-adaptive accuracy on target", target_accs)
    print_stat("Gradual self-train accuracy", final_graduals)
    print_stat("Target self-train accuracy", final_targets)
    print_stat("All self-train accuracy", final_alls)
    print_stat("Best of Target self-train accuracies", best_targets)
    print_stat("Best of All self-train accuracies", best_alls)

# =============================================================================
# 7. Main Execution
# =============================================================================

if __name__ == "__main__":
    SAVE_FILE = 'saved_files/rot_mnist_60_conv_pytorch.pkl'
    
    # Run the experiment
    run_experiment(
        dataset_func=temp,
        n_classes=NUM_CLASSES,
        save_file=SAVE_FILE,
        model_class=SimpleConvNet,
        interval=INTERVAL,
        epochs=EPOCHS,
        loss='ce',
        soft=False,  # Set to True for soft-label self-training
        conf_q=0.1,
        num_runs=NUM_RUNS
    )

    # Display the results
    experiment_results(SAVE_FILE)
