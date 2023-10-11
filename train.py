import os
from datetime import datetime

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras

import dataset


# function for calculation of the accuracy
def calculate_accuracy(logits, gt_logits):  # model output and truth labels
    # apply softmax to convert model output to probabilities
    probabilities = tf.nn.softmax(logits)

    # get the predicted class labels
    predicted_labels = np.argmax(probabilities, axis=1)

    # get the ground truth class labels
    ground_truth_labels = np.argmax(gt_logits, axis=1)

    # calculate accuracy
    accuracy = np.mean(predicted_labels == ground_truth_labels)

    return accuracy


# set the GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# make sure the GPU resources are available
gpus = tf.config.list_physical_devices("GPU")
if not gpus:
    print("No GPUs found. Switching to CPU.")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    tf.config.list_physical_devices("GPU")
else:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# create logs dir
log_dir = "E:\\Licenta\\logs"

num_classes = 7

# batch size
batch_size = 64

# size of the input images
target_size = (75, 100)

print("Loading dataset...")
ds_train, ds_test = dataset.load_dataset(batch_size, target_size)
print("Finished loading dataset.")

# load the pre-trained ResNet50 model without the top classification layer
base_model = keras.applications.ResNet50V2(weights='imagenet', include_top=False,
                                           input_shape=(target_size[0], target_size[1], 3))

# freeze the weights of the pre-trained layers
base_model.trainable = False

# create a new model on top of the pre-trained base model
model = keras.models.Sequential()
model.add(base_model)
model.add(tf.keras.layers.Conv2D(128, (3, 3), activation=tfa.activations.gelu, kernel_initializer='he_uniform',
                                 padding='same'))
model.add(tf.keras.layers.MaxPooling2D((2, 2), padding='same'))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation=tfa.activations.gelu, kernel_initializer='he_uniform',
                                 padding='same'))
model.add(tf.keras.layers.MaxPooling2D((2, 2), padding='same'))
model.add(tf.keras.layers.Conv2D(32, (3, 3), activation=tfa.activations.gelu, kernel_initializer='he_uniform',
                                 padding='same'))
model.add(tf.keras.layers.MaxPooling2D((2, 2), padding='same'))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dense(128, activation=tfa.activations.gelu))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dense(11, activation=tfa.activations.gelu))
model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))

loss_fn = tf.keras.losses.BinaryCrossentropy()
accuracy_metric = tf.keras.metrics.BinaryAccuracy()

logdir = os.path.join(log_dir, datetime.now().strftime("%Y%m%d-%H%M%S"))
file_writer = tf.summary.create_file_writer(os.path.join(logdir, "metrics"))
file_writer.set_as_default()

optimizer = keras.optimizers.RMSprop(learning_rate=1e-3)
num_iters = 10 ** 4
train_iter = iter(ds_train)
test_iter = iter(ds_test)

for step in range(num_iters):
    images, gt_logits = next(train_iter)
    with tf.GradientTape() as tape:
        logits = model(images, training=True)
        loss_value = loss_fn(logits, gt_logits)

    grads = tape.gradient(loss_value, model.trainable_weights)

    # run gradient descent by updating the value of the variables to minimize the loss
    optimizer.apply_gradients(zip(grads, model.trainable_weights))

    # calculate accuracy
    binary_predictions = tf.cast(tf.greater(logits, 0.5), tf.float32)
    accuracy_metric.update_state(gt_logits, binary_predictions)
    accuracy_value = accuracy_metric.result().numpy()
    accuracy_metric.reset_states()

    # log every batch.
    print("Training loss at step %d: %.4f - Accuracy: %.2f%%" % (step, loss_value, accuracy_value * 100))
    tf.summary.scalar("loss", data=loss_value, step=step)
    tf.summary.scalar("accuracy", data=accuracy_value, step=step)

    # at each 100 images, run the model on the validation set  without training it on them
    if step % 100 == 0:
        images, gt_logits = next(test_iter)
        val_logits = model(images, training=False)
        val_loss_value = loss_fn(val_logits, gt_logits)
        val_accuracy_value = calculate_accuracy(val_logits, gt_logits)
        tf.summary.scalar("val_loss", data=val_loss_value, step=step)
        tf.summary.scalar("val_accuracy", data=val_accuracy_value, step=step)
        print("Validation loss: %.4f - Accuracy: %.2f%%" % (val_loss_value, val_accuracy_value * 100))

save_dir = "/models"

# save the trained model
model.save(os.path.join(save_dir, "trained_model_1.h5"))
print("Model saved successfully.")

# evaluate the model on the test dataset
print("Testing the model on the test dataset...")
num_test_steps = 10  # set the number of test steps

test_losses = []
test_accuracies = []
all_predicted_labels = []  # to store all predicted labels
all_ground_truth_labels = []  # to store all ground truth labels

for step, (images, gt_logits) in enumerate(ds_test):
    if step >= num_test_steps:
        break

    logits = model(images, training=False)
    test_loss_value = loss_fn(logits, gt_logits)

    binary_predictions = tf.cast(tf.greater(logits, 0.5), tf.float32)
    accuracy_metric.update_state(gt_logits, binary_predictions)
    test_accuracy_value = accuracy_metric.result().numpy()
    accuracy_metric.reset_states()

    test_losses.append(test_loss_value)
    test_accuracies.append(test_accuracy_value)

    # calculate the predicted labels and ground truth labels for this batch
    predicted_labels = np.argmax(logits, axis=1)
    ground_truth_labels = np.argmax(gt_logits, axis=1)

    all_predicted_labels.extend(predicted_labels)
    all_ground_truth_labels.extend(ground_truth_labels)

    # log every batch for test data as well
    print("Testing loss at step %d: %.4f - Accuracy: %.2f%%" % (step, test_loss_value, test_accuracy_value * 100))
    tf.summary.scalar("test_loss", data=test_loss_value, step=step)
    tf.summary.scalar("test_accuracy", data=test_accuracy_value, step=step)

# calculate the average test loss and accuracy
test_loss = np.mean(test_losses)
test_accuracy = np.mean(test_accuracies)

print("Average Test Loss: %.4f" % (test_loss))
print("Average Test Accuracy: %.2f%%" % (test_accuracy * 100))
