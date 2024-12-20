import tensorflow as tf
from utils.NT_Xent import NT_Xent_tf
from utils.evaluation_metrics import get_top_k_accuracy
from utils.data_augmentations import augment_image
import gc
from utils.test import test

def train_step(X, labels, model, temperature, optimizer, augment_functions):

    with tf.GradientTape() as tape:

        # Perform Data Augmentation
        X_augmented_1 = tf.map_fn(lambda x: augment_image(x, target_size=(224, 224), target_augmentations=augment_functions), X)
        X_augmented_2 = tf.map_fn(lambda x: augment_image(x, target_size=(224, 224), target_augmentations=augment_functions), X)

        # Compute Outputs
        Z_1 = model(X_augmented_1)
        Z_2 = model(X_augmented_2)

        # Calculate Loss
        loss = NT_Xent_tf(Z_1, Z_2, temperature)

    # Compute and apply gradients
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    ## Delete unnecessary variables to save RAM space
    del X_augmented_1, X_augmented_2, Z_1, Z_2, gradients
    gc.collect()

    ## Evaluate
    epoch_top_1_mean_acc_step = get_top_k_accuracy(X, labels, model, k=1, print_acc=False)
    epoch_top_5_mean_acc_step = get_top_k_accuracy(X, labels, model, k=5, print_acc=False)

    return model, optimizer, loss, epoch_top_1_mean_acc_step, epoch_top_5_mean_acc_step



def train(epochs, train_dataset, model, optimizer, temperature, test_dataset=None, augment_functions=['random_crop_and_resize', 'horizontal_flip', 'color_distortion', 'gaussian_blur']):

    # Initialize losses list to track loss
    losses = []
    top_1_acuracies = []
    top_5_acuracies = []
    test_acuracies = []
    train_dataset_count = len(list(train_dataset.as_numpy_iterator()))

    # Training Loop
    for epoch in range(epochs):
        epoch_losses = []
        epoch_top_1_mean_acc = 0
        epoch_top_5_mean_acc = 0

        for X, labels in train_dataset:

            # train for one step
            model, optimizer, loss, epoch_top_1_mean_acc_step, epoch_top_5_mean_acc_step = \
            train_step(X, labels, model, temperature, optimizer, augment_functions=augment_functions)

            # Track losses
            epoch_losses.append(loss)

            # Trach Evaluations
            epoch_top_1_mean_acc += epoch_top_1_mean_acc_step
            epoch_top_5_mean_acc += epoch_top_5_mean_acc_step

        # Normalize Top-1 Accuracies
        epoch_top_1_mean_acc /= train_dataset_count
        epoch_top_5_mean_acc /= train_dataset_count

        # Compute and print average epoch loss
        avg_epoch_loss = tf.reduce_mean(epoch_losses)
        losses.append(avg_epoch_loss)
        top_1_acuracies.append(epoch_top_1_mean_acc)
        top_5_acuracies.append(epoch_top_5_mean_acc)
        if test_dataset is not None:
            test_acc = test(model, test_dataset, k=1, print_results=True, return_all_accs=False)
            test_acuracies.append(test_acc)
        print(f"Epoch {epoch+1}, Average Loss: {avg_epoch_loss.numpy():.4f}, Top 1 Mean Accuracy: {epoch_top_1_mean_acc:.4f}, Top 5 Mean Accuracy: {epoch_top_5_mean_acc:.4f}")

    return model, optimizer, losses, top_1_acuracies, top_5_acuracies, test_acuracies
