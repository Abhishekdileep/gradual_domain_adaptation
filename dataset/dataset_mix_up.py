import collections
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.datasets import mnist
from tensorflow.keras.datasets import cifar10
from tensorflow.random import gamma as tf_random_gamma
import tqdm 
from sklearn.model_selection import train_test_split
import scipy.io
from scipy import ndimage
from scipy.stats import ortho_group
import sklearn.preprocessing
import pickle
from tqdm import tqdm
import utils
import ipdb
import matplotlib.pyplot as plt



def shuffle(xs, ys):
    indices = list(range(len(xs)))
    np.random.shuffle(indices)
    return xs[indices], ys[indices]

def get_preprocessed_mnist():
    (train_x, train_y), (test_x, test_y) = mnist.load_data()
    train_x, test_x = train_x / 255.0, test_x / 255.0
    train_x, train_y = shuffle(train_x, train_y)
    train_x = np.expand_dims(np.array(train_x), axis=-1)
    test_x = np.expand_dims(np.array(test_x), axis=-1)
    return (train_x, train_y), (test_x, test_y)

def continually_rotate_images(xs, start_angle, end_angle):
    new_xs = []
    num_points = xs.shape[0]
    for i in range(num_points):
        angle = float(end_angle - start_angle) / num_points * i + start_angle
        img = ndimage.rotate(xs[i], angle, reshape=False)
        new_xs.append(img)
    return np.array(new_xs)

def sample_rotate_images(xs, start_angle, end_angle):
    new_xs = []
    num_points = xs.shape[0]
    for i in range(num_points):
        if start_angle == end_angle:
            angle = start_angle
        else:
            angle = np.random.uniform(low=start_angle, high=end_angle)
        img = ndimage.rotate(xs[i], angle, reshape=False)
        new_xs.append(img)
    return np.array(new_xs)

def _transition_rotation_dataset(train_x, train_y, test_x, test_y,
                                 source_angles, target_angles, inter_func,
                                 src_train_end, src_val_end, inter_end, target_end):
    assert(target_end <= train_x.shape[0])
    assert(train_x.shape[0] == train_y.shape[0])
    src_tr_x, src_tr_y = train_x[:src_train_end], train_y[:src_train_end]
    src_tr_x = sample_rotate_images(src_tr_x, source_angles[0], source_angles[1])
    src_val_x, src_val_y = train_x[src_train_end:src_val_end], train_y[src_train_end:src_val_end]
    src_val_x = sample_rotate_images(src_val_x, source_angles[0], source_angles[1])
    tmp_inter_x, inter_y = train_x[src_val_end:inter_end], train_y[src_val_end:inter_end]
    inter_x = inter_func(tmp_inter_x)
    dir_inter_x = sample_rotate_images(tmp_inter_x, target_angles[0], target_angles[1])
    dir_inter_y = np.array(inter_y)
    assert(inter_x.shape == dir_inter_x.shape)
    trg_val_x, trg_val_y = train_x[inter_end:target_end], train_y[inter_end:target_end]
    trg_val_x = sample_rotate_images(trg_val_x, target_angles[0], target_angles[1])
    trg_test_x, trg_test_y = test_x, test_y
    trg_test_x = sample_rotate_images(trg_test_x, target_angles[0], target_angles[1])
    return (src_tr_x, src_tr_y, src_val_x, src_val_y, inter_x, inter_y,
            dir_inter_x, dir_inter_y, trg_val_x, trg_val_y, trg_test_x, trg_test_y)

def make_rotated_dataset(train_x, train_y, test_x, test_y,
                         source_angles, inter_angles, target_angles,
                         src_train_end, src_val_end, inter_end, target_end):
    inter_func = lambda x: continually_rotate_images(x, inter_angles[0], inter_angles[1])
    return _transition_rotation_dataset(
        train_x, train_y, test_x, test_y, source_angles, target_angles,
        inter_func, src_train_end, src_val_end, inter_end, target_end)

def rotated_mnist_60_data_func(src_tr_end=5000, src_val_end=6000, inter_end=48000, target_end=50000):
    (train_x, train_y), (test_x, test_y) = get_preprocessed_mnist()
    return make_rotated_dataset(
        train_x, train_y, test_x, test_y, [0.0, 5.0], [5.0, 60.0], [55.0, 60.0],
        src_tr_end, src_val_end, inter_end, target_end)

def sample_beta_distribution(size, concentration_0=0.2, concentration_1=0.2):
    gamma_1_sample = tf_random_gamma(shape=[size], alpha=concentration_1)
    gamma_2_sample = tf_random_gamma(shape=[size], alpha=concentration_0)
    return gamma_1_sample / (gamma_1_sample + gamma_2_sample)

def sort_dataset(train_source_x , train_source_y):
	sorted_indices_source = np.argsort(train_source_y)
	sorted_train_source_x = train_source_x[sorted_indices_source]
	sorted_train_source_y = train_source_y[sorted_indices_source]
	count_source = dict()
	for i in np.unique(train_source_y):
		count_source[i.item()] = 0 
	for i in train_source_y:
		count_source[i.item()] += 1  

	return count_source , sorted_train_source_x, sorted_train_source_y
  

def mix_up(train_source_x, train_source_y, train_tgt_x, train_tgt_y, alpha):
    appended_x = []
    appended_y = []
    
    ################################################
    #####Adding the Mixup class wise using sort ####  
    ################################################
    
    count_source , sorted_train_source_x, sorted_train_source_y = sort_dataset(train_source_x , train_source_y)
    count_tgt , sorted_train_tgt_x, sorted_train_tgt_y = sort_dataset(train_tgt_x , train_tgt_y)
    
    min_count = dict()
    previous_source_count = 0
    previous_tgt_count = 0
    import ipdb
    ipdb.set_trace()
    assert set(count_source.keys()) == set(count_tgt.keys()), "Source and target datasets must have the same classes."
    cropped_train_source_x = []
    cropped_train_source_y = []
    cropped_train_tgt_x = []
    cropped_train_tgt_y = []
    for i in count_source.keys():
        min_count[i] = min(count_source[i], count_tgt[i])
        cropped_train_source_x.append(sorted_train_source_x[previous_source_count:previous_source_count + min_count[i]])
        cropped_train_source_y.append(sorted_train_source_y[previous_source_count:previous_source_count + min_count[i]])
        cropped_train_tgt_x.append(sorted_train_tgt_x[previous_tgt_count:previous_tgt_count + min_count[i]])
        cropped_train_tgt_y.append(sorted_train_tgt_y[previous_tgt_count:previous_tgt_count + min_count[i]])
        previous_source_count += count_source[i]
        previous_tgt_count += count_tgt[i] 
    
    
    batch_size = min(sorted_train_source_x.shape[0], sorted_train_tgt_x.shape[0])
    for i in tqdm(alpha, desc="Mixing up datasets"):
        l = sample_beta_distribution(batch_size, i, i).numpy()
        x_l = l.reshape((batch_size, 1, 1, 1))
        y_l = l.reshape((batch_size,))  # Corrected shape for y_l
        mixed_x = x_l * sorted_train_source_x[:batch_size] + (1 - x_l) * sorted_train_tgt_x[:batch_size]
        mixed_y = (y_l * sorted_train_source_y[:batch_size] + (1 - y_l) * sorted_train_tgt_y[:batch_size]).astype(int)  # Ensure integer labels
        appended_x.append(mixed_x)
        appended_y.append(mixed_y)

    ipdb.set_trace()
    
    appended_x , appended_y = np.concatenate(appended_x, axis=0), np.concatenate(appended_y, axis=0)
    return appended_x, appended_y

def rotated_mnist_mix_up_data_func():
    (src_tr_x, src_tr_y, src_val_x, src_val_y, inter_x, inter_y, dir_inter_x, dir_inter_y,
        trg_val_x, trg_val_y, trg_test_x, trg_test_y) = rotated_mnist_60_data_func(20000, 22000, 48000, 50000)
    inter_x , inter_y = mix_up(src_tr_x , src_tr_y , dir_inter_x, dir_inter_y, alpha=[0.2 , 0.4 , 0.6 , 0.8 ])
    return (src_tr_x, src_tr_y, src_val_x, src_val_y, inter_x, inter_y, dir_inter_x, dir_inter_y,
        trg_val_x, trg_val_y, trg_test_x, trg_test_y)

def save_image(train_x , train_y , file_number=0):
    train_x = train_x * 255.0
    img = train_x[file_number].reshape(28, 28)
    plt.imsave(f'saved_images/mnist_{file_number}.png' , img, cmap='gray') 
   
if __name__ == "__main__":
    rotated_mnist_mix_up_data_func()