import numpy as np 
from dataset_mix_up import rotated_mnist_60_data_func

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