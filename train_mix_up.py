import models
import dataset_mix_up as datasets
import numpy as np
import tensorflow as tf
from tensorflow.keras import metrics
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import pickle
import ipdb

####################### Utils.py ########################

def rand_seed(seed):
    ipdb.set_trace()
    np.random.seed(seed)
    tf.compat.v1.set_random_seed(seed)

def soft_self_train_once(student, teacher, unsup_x, epochs=20):
    ipdb.set_trace()
    probs = teacher.predict(np.concatenate([unsup_x]))
    student.fit(unsup_x, probs, epochs=epochs, verbose=False)

def self_train_once(student, teacher, unsup_x, confidence_q=0.1, epochs=20):
    # Do one bootstrapping step on unsup_x, where pred_model is used to make predictions,
    # and we use these predictions to update model.
    ipdb.set_trace()
    logits = teacher.predict(np.concatenate([unsup_x]))
    confidence = np.amax(logits, axis=1) - np.amin(logits, axis=1)
    alpha = np.quantile(confidence, confidence_q)
    indices = np.argwhere(confidence >= alpha)[:, 0]
    preds = np.argmax(logits, axis=1)
    student.fit(unsup_x[indices], preds[indices], epochs=epochs, verbose=False)
    
def self_train(student_func, teacher, unsup_x, confidence_q=0.1, epochs=20, repeats=1,
               target_x=None, target_y=None, soft=False):
    ipdb.set_trace()
    accuracies = []
    for i in range(repeats):
        student = student_func(teacher)
        if soft:
            soft_self_train_once(student, teacher, unsup_x, epochs)
        else:
            self_train_once(student, teacher, unsup_x, confidence_q, epochs)
        if target_x is not None and target_y is not None:
            _, accuracy = student.evaluate(target_x, target_y, verbose=True)
            accuracies.append(accuracy)
        teacher = student
    return accuracies, student

########################## Gradual Self-Training ##########################

def gradual_self_train(student_func, teacher, unsup_x, debug_y, interval, confidence_q=0.1,epochs=20, soft=False , upper_idx=None):
    ipdb.set_trace()
    if upper_idx is not None:
        upper_idx = int(unsup_x.shape[0] / interval)
    accuracies = []
    for i in range(upper_idx):
        student = student_func(teacher)
        cur_xs = unsup_x[interval*i:interval*(i+1)]
        cur_ys = debug_y[interval*i:interval*(i+1)]
        # _, student = self_train(
        #     student_func, teacher, unsup_x, confidence_q, epochs, repeats=2)
        if soft:
            soft_self_train_once(student, teacher, cur_xs, epochs)
        else:
            self_train_once(student, teacher, cur_xs, confidence_q, epochs)
        _, accuracy = student.evaluate(cur_xs, cur_ys)
        accuracies.append(accuracy)
        teacher = student
    return accuracies, student

def compile_model(model, loss='ce'):
    ipdb.set_trace()
    loss = models.get_loss(loss, model.output_shape[1])
    model.compile(optimizer='adam',
                  loss=[loss],
                  metrics=[metrics.sparse_categorical_accuracy])

def run_experiment(
    dataset_func, n_classes, input_shape, save_file, model_func=models.simple_softmax_conv_model, interval=2000, epochs=10, loss='ce', 
    soft=False, conf_q=0.1, num_runs=20, num_repeats=None , find_upper_idx=False):
    (src_tr_x, src_tr_y, src_val_x, src_val_y, # Source training and validation datset  
    inter_x, inter_y, dir_inter_x, dir_inter_y,# intermediate unsupervised data and labels
    trg_val_x, trg_val_y, trg_test_x, trg_test_y) = dataset_func() 
    
    print(f'Source training data shape: {src_tr_x.shape}, Source training labels shape: {src_tr_y.shape}')
    print(f'Source validation data shape: {src_val_x.shape}, Source validation labels shape: {src_val_y.shape}')
    print(f'Intermediate unsupervised data shape: {inter_x.shape}, Intermediate unsupervised labels shape: {inter_y.shape}')
    print(f'Direct bootstrap (target) data shape: {dir_inter_x.shape}, Direct bootstrap (target) labels shape: {dir_inter_y.shape}') 
    print(f'Target validation data shape: {trg_val_x.shape}, Target validation labels shape: {trg_val_y.shape}')
    print(f'Target test data shape: {trg_test_x.shape}, Target test labels shape: {trg_test_y.shape}')
    upper_idx = None
    if find_upper_idx:
        ipdb.set_trace()
        upper_idx = min(src_tr_x.shape[0], dir_inter_x.shape[0])
    #converting to categorical one hot encoding if soft is True
    if soft:
        src_tr_y = to_categorical(src_tr_y)
        src_val_y = to_categorical(src_val_y)
        trg_eval_y = to_categorical(trg_eval_y)
        dir_inter_y = to_categorical(dir_inter_y)
        inter_y = to_categorical(inter_y)
        trg_test_y = to_categorical(trg_test_y)
        
    if num_repeats is None:
        num_repeats = int(inter_x.shape[0] / interval)
    print(f'Number of repeats: {num_repeats}\nNumber of runs: {num_runs}\nsoft training: {soft}\n'
          f' Confidence thresholding: {conf_q}\n Interval: {interval}\n Epochs: {epochs}\n Loss: {loss}')
    
    def new_model():
        model = model_func(n_classes, input_shape=input_shape)
        compile_model(model, loss)
        return model
    def student_func(teacher):
        return teacher
    def run(seed):
        rand_seed(seed)
        trg_eval_x = trg_test_x
        trg_eval_y = trg_test_y
        # Train source model.
        source_model = new_model()
        print("\n\n Source Model Training :")
        source_model.fit(src_tr_x, src_tr_y, epochs=epochs, verbose=False)
        _, src_acc = source_model.evaluate(src_val_x, src_val_y)
        _, target_acc = source_model.evaluate(trg_eval_x, trg_eval_y)
        # Gradual self-training.
        
        with ipdb.launch_ipdb_on_exception():
            ############## GDA #####################
            print("\n\n Gradual self-training:")
            teacher = new_model()
            teacher.set_weights(source_model.get_weights())
            gradual_accuracies, student = gradual_self_train(
                student_func, teacher, inter_x, inter_y, interval, epochs=epochs, soft=soft,
                confidence_q=conf_q , upper_idx=upper_idx)
            _, acc = student.evaluate(trg_eval_x, trg_eval_y)
            gradual_accuracies.append(acc)
           
           ###################### Direct bootstrap #####################
            print("\n\n Direct boostrap to target:")
            teacher = new_model()
            teacher.set_weights(source_model.get_weights())
            target_accuracies, _ = self_train(
                student_func, teacher, dir_inter_x, epochs=epochs, target_x=trg_eval_x,
                target_y=trg_eval_y, repeats=num_repeats, soft=soft, confidence_q=conf_q)
            
            ####################### Direct bootstrap to all unsup data #####################
            print("\n\n Direct boostrap to all unsup data:")
            teacher = new_model()
            teacher.set_weights(source_model.get_weights())
            all_accuracies, _ = self_train(
                student_func, teacher, inter_x, epochs=epochs, target_x=trg_eval_x,
                target_y=trg_eval_y, repeats=num_repeats, soft=soft, confidence_q=conf_q)
        return src_acc, target_acc, gradual_accuracies, target_accuracies, all_accuracies
    
    results = []
    for i in range(num_runs):
        print(f'Run {i + 1}/{num_runs}')
        results.append(run(i))
    print('Saving to ' + save_file)
    pickle.dump(results, open(save_file, "wb"))

def mixup_mnist_60_conv_experiment():
    ipdb.set_trace()
    run_experiment(
        dataset_func=datasets.rotated_mnist_mix_up_data_func, 
        n_classes=10, 
        input_shape=(28, 28, 1),
        save_file='saved_files/mixup_mnist_60_conv.txt',
        model_func=models.simple_softmax_conv_model, 
        interval=20000, 
        epochs=10, 
        loss='ce',
        soft=False, 
        conf_q=0.1, 
        num_runs=5 , 
        find_upper_idx=True)
    
def experiment_results(save_name):
    ipdb.set_trace()
    results = pickle.load(open(save_name, "rb"))
    src_accs, target_accs = [], []
    final_graduals, final_targets, final_alls = [], [], []
    best_targets, best_alls = [], []
    for src_acc, target_acc, gradual_accuracies, target_accuracies, all_accuracies in results:
        src_accs.append(100 * src_acc)
        target_accs.append(100 * target_acc)
        final_graduals.append(100 * gradual_accuracies[-1])
        final_targets.append(100 * target_accuracies[-1])
        final_alls.append(100 * all_accuracies[-1])
        best_targets.append(100 * np.max(target_accuracies))
        best_alls.append(100 * np.max(all_accuracies))
    num_runs = len(src_accs)
    mult = 1.645  # For 90% confidence intervals
    print("\nNon-adaptive accuracy on source (%): ", np.mean(src_accs),
          mult * np.std(src_accs) / np.sqrt(num_runs))
    print("Non-adaptive accuracy on target (%): ", np.mean(target_accs),
          mult * np.std(target_accs) / np.sqrt(num_runs))
    print("Gradual self-train accuracy (%): ", np.mean(final_graduals),
          mult * np.std(final_graduals) / np.sqrt(num_runs))
    print("Target self-train accuracy (%): ", np.mean(final_targets),
          mult * np.std(final_targets) / np.sqrt(num_runs))
    print("All self-train accuracy (%): ", np.mean(final_alls),
          mult * np.std(final_alls) / np.sqrt(num_runs))
    print("Best of Target self-train accuracies (%): ", np.mean(best_targets),
          mult * np.std(best_targets) / np.sqrt(num_runs))
    print("Best of All self-train accuracies (%): ", np.mean(best_alls),
          mult * np.std(best_alls) / np.sqrt(num_runs))
    
if __name__ == "__main__":
    ipdb.set_trace()

    mixup_mnist_60_conv_experiment()
    print("Mixup MNIST conv experiment")
    experiment_results('saved_files/mixup_mnist_60_conv.txt')

       