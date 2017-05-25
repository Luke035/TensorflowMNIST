from tensorflow.contrib import learn
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib import learn

ROOT_DATA_PATH = '/var/ifs/data/hadoop-cloudera5/notebookDir/HUB/lgrazioli/Data/'
TRAIN_FILE_NAME = 'train.csv'
TEST_FILE_NAME = 'test.csv'
MODEL_CHECKPOINT_DIR = "/tmp/mnist_model"
SUBMISSION_FILE = "'/tmp/digit_submission2.csv'"

def max_pool_2x2(tensor_in):
	return tf.nn.max_pool(tensor_in, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def reshapeDataframe(toReshapeDf, rowDim1, rowDim2):
        data_frame_size = len(toReshapeDf)
        #Must be casted to np.float32
        return toReshapeDf.values.reshape(data_frame_size, rowDim1, rowDim2, 1).astype(np.float32)

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


#Retrieve test set as panda_df
train_df = pd.read_csv(ROOT_DATA_PATH + TRAIN_FILE_NAME).sample(n=1, replace=False, axis=0)
data_df = train_df[train_df.columns[1:]]
label_df = train_df[train_df.columns[0]]


#Restore last checkpoint (must resotre through fit with 0 steps https://github.com/tensorflow/tensorflow/issues/3340)
classifier = learn.Estimator(model_fn=my_model, model_dir=MODEL_CHECKPOINT_DIR)
classifier.fit(reshapeDataframe(data_df, 28, 28), label_df , steps=0)

#Load test dataframe
test_df = pd.read_csv(ROOT_DATA_PATH + TEST_FILE_NAME)
#Predict the test dataframe
predictionList = [i['class'] for i in list(classifier.predict(reshapeDataframe(test_df, 28, 28), batch_size=256))]

#Manipulate DATAFRAME in oprder to satisfy KAGGLE requirements
submission_pd = pd.DataFrame(predictionList, columns=['Label'])
submission_pd['ImageId'] = range(1, len(submission_pd) + 1)

to_submit_pd = pd.DataFrame(submission_pd['ImageId'])
to_submit_pd['Label'] = submission_pd['Label']

to_submit_pd.to_csv(path_or_buf=SUBMISSION_FILE, sep=',', index=False)