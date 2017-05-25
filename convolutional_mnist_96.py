from sklearn import metrics, cross_validation
import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib import learn
import pandas as pd
import numpy as np
import time

### Enable Training step logging
tf.logging.set_verbosity(tf.logging.INFO)

#Global variables definition
ROOT_DATA_PATH = '/var/ifs/data/hadoop-cloudera5/notebookDir/HUB/lgrazioli/Data/'
TRAIN_FILE_NAME = 'train.csv'
TEST_FILE_NAME = 'test.csv'
MODEL_EXPORT_PATH = '/var/ifs/data/hadoop-cloudera5/notebookDir/HUB/lgrazioli/Data/modelExport'
MODEL_CHECKPOINT_DIR = "/tmp/mnist_model"
#Set to TRUE if you want to SAMPLE the trainig set
SAMPLE = False

#Utility function to rehsape training set
def reshapeDataframe(toReshapeDf, rowDim1, rowDim2):
        data_frame_size = len(toReshapeDf)
        #Must be casted to np.float32
        return toReshapeDf.values.reshape(data_frame_size, rowDim1, rowDim2, 1).astype(np.float32)

#Utility function to wrap a 2x2 max-pooling operation
def max_pool_2x2(tensor_in):
	return tf.nn.max_pool(tensor_in, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

'''
Model definition
MNIST model is composed as follows:
- conv_layer1: (28,28,32) output shape with kernel_window 5x5 and strides (1,1). Activation function: RELU
- max_pool: (14,14,32) output shape 2x2 max_pooling
- conv_layer2: (14,14,64) output shape with kernel_window 5x5 and strides (1,1). Activation function: RELU
- max_pool: (7,7,64) output shape 2x2 max_pooling
- reshape: (?, 3136), derived by (7*7*64)
- fully_connected_layer: (?, 1024) ouput shape. Activation function: RELU
- droput
- readout_layer: (?, 10) output shape
'''
def my_model(features, target):
        '''
        one-hot Porta il vettore target a una matrice one-hot
        es: [2,1,0,8]
        [
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
        ]
        '''
        print(features.shape)
        target = tf.one_hot(tf.cast(target, tf.int32), 10, 1, 0)
        #Stacking 2 fully connected layers
        #features = layers.stack(features, layers.fully_connected, [100, 10])

        #First convolutional layer
        with tf.variable_scope('conv_layer1'):
            features = layers.convolution2d(inputs=features, num_outputs=32, kernel_size=[5,5], data_format='NHWC', activation_fn=tf.nn.relu)
            features = max_pool_2x2(features)
        #Ogni immagine esce con un numero di canali uguale al num_outputs
        #(28, 28, 1) -> (28, 28, 100)
        
        #Second convolutional layer
        with tf.variable_scope('conv_layer2'):
            features = layers.convolution2d(inputs=features, num_outputs=64, kernel_size=[5,5], data_format='NHWC', activation_fn=tf.nn.relu)
            features = max_pool_2x2(features)

        # Back to bidimensional space
        features = tf.reshape(features, [- 1, 64 * 7 * 7])

        #Fully connected layer
        features = layers.fully_connected(features, 1024, activation_fn=tf.nn.relu)

        #Dropout
        features = layers.dropout(features, keep_prob=0.5, is_training=True)

        #Readout layer
        features = layers.fully_connected(features, 10, activation_fn=None)

        #Batch prediction and loss function
        prediction, loss = (tf.contrib.learn.models.logistic_regression_zero_init(features, target))

        #Training optimizer
        train_op = tf.contrib.layers.optimize_loss(
                loss, tf.contrib.framework.get_global_step(),
                optimizer='SGD',learning_rate=0.001
        )
        return {'class': tf.argmax(prediction, 1), 'prob': prediction}, loss, train_op

'''
TENSORFLOW model
'''
def conv_model(feature, target, mode):
    """2-layer convolution model."""
    # Convert the target to a one-hot tensor of shape (batch_size, 10) and
    # with a on-value of 1 for each one-hot vector of length 10.
    target = tf.one_hot(tf.cast(target, tf.int32), 10, 1, 0)

    # Reshape feature to 4d tensor with 2nd and 3rd dimensions being
    # image width and height final dimension being the number of color channels.
    #feature = tf.reshape(feature, [-1, 28, 28, 1])
    # First conv layer will compute 32 features for each 5x5 patch
    with tf.variable_scope('conv_layer1'):
        h_conv1 = layers.convolution2d(feature, 32, kernel_size=[5, 5], activation_fn=tf.nn.relu)
        h_pool1 = max_pool_2x2(h_conv1)

    # Second conv layer will compute 64 features for each 5x5 patch.
    with tf.variable_scope('conv_layer2'):
        h_conv2 = layers.convolution2d(h_pool1, 64, kernel_size=[5, 5], activation_fn=tf.nn.relu)
        h_pool2 = max_pool_2x2(h_conv2)
        # reshape tensor into a batch of vectors
        h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])

    # Densely connected layer with 1024 neurons.
    h_fc1 = layers.dropout(layers.fully_connected(h_pool2_flat, 1024, activation_fn=tf.nn.relu),
        keep_prob=0.5,
        is_training=mode == tf.contrib.learn.ModeKeys.TRAIN)
    
    # Compute logits (1 per class) and compute loss.
    logits = layers.fully_connected(h_fc1, 10, activation_fn=None)
    loss = tf.losses.softmax_cross_entropy(target, logits)

    # Create a tensor for training op.
    train_op = layers.optimize_loss(
        loss,
        tf.contrib.framework.get_global_step(),
        optimizer='SGD',
        learning_rate=0.001)

    return {'class': tf.argmax(logits, 1), 'prob': logits}, loss, train_op

# Dataset is read as PANDA dataframe, if SAMPLE is TRUE is sampled
if SAMPLE:  
        train_df = pd.read_csv(ROOT_DATA_PATH + TRAIN_FILE_NAME).sample(frac=0.10, replace=False, axis=0)
else:
        train_df = pd.read_csv(ROOT_DATA_PATH + TRAIN_FILE_NAME)

#Split in data columns and training columns
data_df = train_df[train_df.columns[1:]]
label_df = train_df[train_df.columns[0]]

#Sckit learn splitting methods
x_train, x_test, y_train, y_test = cross_validation.train_test_split(data_df, label_df, test_size=0.1, random_state=35)

#Utility code to measure training time
start_time = time.time()

print("Training started at "+str(start_time))

#Validation monitor for TENSORBOARD
validation_monitor = tf.contrib.learn.monitors.ValidationMonitor(reshapeDataframe(x_test, 28, 28), y_test.values.astype(np.int64), every_n_steps=100)

#CLASSIFIER definition
classifier = learn.Estimator(model_fn=my_model, config=tf.contrib.learn.RunConfig(save_checkpoints_secs=5) , model_dir=MODEL_CHECKPOINT_DIR)
#CLASSIFIER fit (launch train)
classifier.fit(reshapeDataframe(x_train, 28, 28), y_train, steps=100000, batch_size = 256, monitors=[validation_monitor])

#Utility code to measure training elapsed time
elapsed_time = time.time() - start_time
print("Training completed in : " + str(elapsed_time)+ " seconds")

#Predict validation set and convert as PANDA dataframe
predictionList = [i['class'] for i in list(classifier.predict(reshapeDataframe(x_test, 28, 28)))]

# Measure accuracy
score = metrics.accuracy_score(y_test, predictionList)
print("Test accuracy " +str(score))