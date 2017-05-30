from tensorflow.contrib import learn
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib

ROOT_DATA_PATH = '~/tensorflowMNIST/'
TRAIN_FILE_NAME = 'train.csv'
TEST_FILE_NAME = 'test.csv'
MODEL_CHECKPOINT_DIR = "~/mnist_model_dir"
SUBMISSION_FILE = "~/digit_submission2.csv"

SAMPLE = False         #Set to TRUE if you want to SAMPLE the trainig set
LEARNING_RATE = 0.001
OPTIMIZER = 'SGD'
STEPS = 10000
BATCH_SIZE = 20
CHECKPOINTS_SECS = 30
VALIDATION_STEPS = 500
EPOCHS = 1
model_params = {"learning_rate": LEARNING_RATE, "optimizer": OPTIMIZER}


def max_pool_2x2(tensor_in):
	return tf.nn.max_pool(tensor_in, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def reshapeDataframe(toReshapeDf, rowDim1, rowDim2):
        data_frame_size = len(toReshapeDf)
        #Must be casted to np.float32
        return toReshapeDf.values.reshape(data_frame_size, rowDim1, rowDim2, 1).astype(np.float32)

def my_model(features, target, mode, params):
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
        print(target.shape)
        target = tf.one_hot(tf.cast(target, tf.int32), 10, 1, 0)
        print(target.shape)
        #Stacking 2 fully connected layers
        #features = layers.stack(features, layers.fully_connected, [100, 10])

        #First convolutional layer
        with tf.variable_scope('conv_layer1'):
            features = layers.convolution2d(inputs=features, num_outputs=32, kernel_size=[5,5], data_format='NHWC', activation_fn=tf.nn.relu )
            features = layers.max_pool2d(inputs=features, kernel_size=2, stride=2,padding='SAME', data_format='NHWC' )
        #Ogni immagine esce con un numero di canali uguale al num_outputs
        #(28, 28, 1) -> (28, 28, 100)
        
        #Second convolutional layer
        with tf.variable_scope('conv_layer2'):
            features = layers.convolution2d(inputs=features, num_outputs=64, kernel_size=[5,5], data_format='NHWC', activation_fn=tf.nn.relu )
            features = layers.max_pool2d(inputs=features, kernel_size=2, stride=2,padding='SAME', data_format='NHWC' )
  
        # Back to bidimensional space
        features = tf.reshape(features, [- 1, 64 * 7 * 7])

        #Fully connected layer
        with tf.variable_scope('fc_layer1'):
            features = layers.fully_connected(features, 1024, activation_fn=tf.nn.relu)

        #Dropout
        with tf.variable_scope('dropout'):
            features = layers.dropout(features, keep_prob=0.5, is_training=True)

        #Readout layerinput_fn
        with tf.variable_scope('fc_layer2'):
            features = layers.fully_connected(features, 10, activation_fn=None)

        #Loss function
        with tf.variable_scope('loss'):
            loss = tf.losses.softmax_cross_entropy(target, features)
        
        with tf.variable_scope('train'):
            train_op = tf.contrib.layers.optimize_loss(
                    loss, 
                    tf.contrib.framework.get_global_step(),
                    optimizer = params["optimizer"],
                    learning_rate = params["learning_rate"]
            )

        #Dictionaries
        predictions = {
                "class": tf.argmax(features, 1)
        }
        eval_metric_ops = {"accuracy": tf.metrics.accuracy(tf.argmax(target,1), tf.argmax(features,1))}

        return model_fn_lib.ModelFnOps(
                mode = mode,
                predictions = predictions,
                loss = loss,
                train_op = train_op,
                eval_metric_ops = eval_metric_ops)



#Retrieve test set as panda_df
train_df = pd.read_csv(ROOT_DATA_PATH + TRAIN_FILE_NAME).sample(n=1, replace=False, axis=0)
data_df = train_df[train_df.columns[1:]]
label_df = train_df[train_df.columns[0]]

x_train = reshapeDataframe(data_df, 28, 28)

#Restore last checkpoint (must resotre through fit with 0 steps https://github.com/tensorflow/tensorflow/issues/3340)
classifier = learn.Estimator(model_fn=my_model, params=model_params, model_dir=MODEL_CHECKPOINT_DIR)
classifier.fit(x_train, label_df , steps=0)

#Load test dataframe
test_df = pd.read_csv(ROOT_DATA_PATH + TEST_FILE_NAME)
x_test = reshapeDataframe(test_df, 28, 28)
#Predict the test dataframe
predictionList = [i['class'] for i in list(classifier.predict(x_test, batch_size=256))]

#Manipulate DATAFRAME in oprder to satisfy KAGGLE requirements
submission_pd = pd.DataFrame(predictionList, columns=['Label'])
submission_pd['ImageId'] = range(1, len(submission_pd) + 1)

to_submit_pd = pd.DataFrame(submission_pd['ImageId'])
to_submit_pd['Label'] = submission_pd['Label']

to_submit_pd.to_csv(path_or_buf=SUBMISSION_FILE, sep=',', index=False)
