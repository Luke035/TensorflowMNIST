import tensorflow as tf
from sklearn import cross_validation
from tensorflow.contrib import layers
from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib
import pandas as pd
import numpy as np
import time

### Enable Training step logging
tf.logging.set_verbosity(tf.logging.INFO)

#Global variables definition
ROOT_PATH = '/var/ifs/data/hadoop-cloudera5/notebookDir/HUB/lgrazioli/Data/'
ROOT_DATA_PATH = ROOT_PATH #+ '/data'
TRAIN_FILE_NAME = 'train.csv'
TEST_FILE_NAME = 'test.csv'
MODEL_EXPORT_PATH = '/var/ifs/data/hadoop-cloudera5/notebookDir/HUB/lgrazioli/Data/modelExport'
MODEL_CHECKPOINT_DIR = "/tmp/mnist_model"

SAMPLE = False         #Set to TRUE if you want to SAMPLE the trainig set
LEARNING_RATE = 0.001
OPTIMIZER = 'SGD'
STEPS = 10000
BATCH_SIZE = 256
CHECKPOINTS_SECS = 30
VALIDATION_STEPS = 500
EPOCHS = 1
model_params = {"learning_rate": LEARNING_RATE, "optimizer": OPTIMIZER}

def get_batched_input_fn(x, y, batch_size=BATCH_SIZE, epochs=EPOCHS, seed=35):
    sliced_input = tf.train.slice_input_producer([x, y.values])
    return tf.train.batch(sliced_input, batch_size=batch_size, num_threads=4)

def get_input_fn(x, y):
    return tf.constant(x), tf.constant(y)

#Utility function to rehsape training set
'''def reshapeDataframe(toReshapeDf, rowDim1, rowDim2):
        data_frame_size = len(toReshapeDf)
        #Must be casted to np.float32
        return tf.constant(toReshapeDf.values.reshape(data_frame_size, rowDim1, rowDim2, 1).astype(np.float32))'''

def reshapeDataframe(toReshapeDf, rowDim1, rowDim2):
        data_frame_size = len(toReshapeDf)
        #Must be casted to np.float32
        return toReshapeDf.values.reshape(data_frame_size, rowDim1, rowDim2, 1).astype(np.float32)

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


# Dataset is read as PANDA dataframe, if SAMPLE is TRUE is sampled


if SAMPLE:  
        train_df = pd.read_csv(ROOT_DATA_PATH + TRAIN_FILE_NAME).sample(frac=0.10, replace=False, axis=0)
else:
        train_df = pd.read_csv(ROOT_DATA_PATH + TRAIN_FILE_NAME)

#Split in data columns and training columns
data_df = train_df[train_df.columns[1:]]
label_df = train_df[train_df.columns[0]]

#Sckit learn splitting methods
x_train, x_test, y_train, y_test = cross_validation.train_test_split(data_df, label_df, test_size=0.05, random_state=35)

x_train = reshapeDataframe(x_train, 28, 28)
x_test = reshapeDataframe(x_test, 28, 28)

#Utility code to measure training time
start_time = time.time()

print("Training started at "+str(start_time))

#Validation monitor for TENSORBOARD
#validation_monitor = tf.contrib.learn.monitors.ValidationMonitor(input_fn=lambda:get_input_fn(x_test, y_test), every_n_steps=VALIDATION_STEPS)
validation_monitor = tf.contrib.learn.monitors.ValidationMonitor(x_test, y_test.values.astype(np.int64), every_n_steps=VALIDATION_STEPS)
#Config proto for GPU options
config = tf.ConfigProto(log_device_placement=False)
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 1
 
#CLASSIFIER definition
#Config proto can be used only with TF >= 1.2.0
classifier = learn.Estimator(model_fn=my_model, params=model_params, config=tf.contrib.learn.RunConfig(save_checkpoints_secs=CHECKPOINTS_SECS, session_config=config), model_dir=MODEL_CHECKPOINT_DIR)
#CLASSIFIER fit (launch train)
classifier.fit(input_fn=lambda:get_batched_input_fn(x_train, y_train), steps=STEPS, monitors=[validation_monitor])


#Utility code to measure training elapsed time
elapsed_time = time.time() - start_time
print("Training completed in : " + str(elapsed_time)+ " seconds")

# Score accuracy
ev = classifier.evaluate(input_fn=lambda:get_input_fn(x_test, y_test), steps=1)
print("Loss: %s" % ev["loss"])
print("Accuracy: %s" % ev["accuracy"])
