#!/usr/bin/env python

########################################################################
# import default python-library
########################################################################
import pickle
import os
import sys
import glob
########################################################################


########################################################################
# import additional python-library
########################################################################
import numpy
import librosa
import librosa.core
import librosa.feature
import yaml
import logging
# from import
from tqdm import tqdm
from sklearn import metrics
from keras.models import Model
from keras.layers import Input, Dense
########################################################################



__versions__ = "1.0.3"
########################################################################


########################################################################
# setup STD I/O
########################################################################
"""
Standard output is logged in "baseline.log".
"""

# Set up logging configuration
logging.basicConfig(level=logging.DEBUG, filename="baseline.log")

# Create a logger object
logger = logging.getLogger(' ')

# Create a stream handler to output log messages to the console
handler = logging.StreamHandler()

# Create a formatter to specify the log message format
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

# Set the formatter for the handler
handler.setFormatter(formatter)

# Add the handler to the logger
logger.addHandler(handler)

########################################################################


########################################################################
# visualizer
########################################################################

# Define a class named "visualizer"
class visualizer(object):
    def __init__(self):
        # Import the matplotlib.pyplot module as plt
        import matplotlib.pyplot as plt
        # Create an instance of the plt module and assign it to self.plt
        self.plt = plt
        # Create a new figure with a specified size
        self.fig = self.plt.figure(figsize=(30, 10))
        # Adjust the spacing between subplots
        self.plt.subplots_adjust(wspace=0.3, hspace=0.3)

    def loss_plot(self, loss, val_loss):
        """
        Plot loss curve.

        loss : list [ float ]
            training loss time series.
        val_loss : list [ float ]
            validation loss time series.

        return   : None
        """
        # Add a subplot to the current figure
        ax = self.fig.add_subplot(1, 1, 1)
        # Clear the current axes
        ax.cla()
        # Plot the training loss
        ax.plot(loss)
        # Plot the validation loss
        ax.plot(val_loss)
        # Set the title of the plot
        ax.set_title("Model loss")
        # Set the label for the x-axis
        ax.set_xlabel("Epoch")
        # Set the label for the y-axis
        ax.set_ylabel("Loss")
        # Add a legend to the plot
        ax.legend(["Train", "Test"], loc="upper right")

    def save_figure(self, name):
        """
        Save figure.

        name : str
            save .png file path.

        return : None
        """
        self.plt.savefig(name)


########################################################################


########################################################################
# file I/O
########################################################################
# pickle I/O

# Define a function named "save_pickle" that takes in a filename and save_data as parameters
def save_pickle(filename, save_data):
    """
    picklenize the data.

    filename : str
        pickle filename
    data : free datatype
        some data will be picklenized

    return : None
    """
    # Log an info message indicating the start of saving the pickle file
    logger.info("save_pickle -> {}".format(filename))
    # Open the file with the given filename in write binary mode
    with open(filename, 'wb') as sf:
        # Pickle the save_data and write it to the file
        pickle.dump(save_data, sf)



# Define a function named "load_pickle" that takes in a filename as a parameter
def load_pickle(filename):
    """
    unpicklenize the data.

    filename : str
        pickle filename

    return : data
    """
    # Log an info message indicating the start of loading the pickle file
    logger.info("load_pickle <- {}".format(filename))
    # Open the file with the given filename in read binary mode
    with open(filename, 'rb') as lf:
        # Unpickle the data from the file and assign it to the variable "load_data"
        load_data = pickle.load(lf)
    # Return the unpickled data
    return load_data



# wav file Input

# Define a function named "file_load" that takes in parameters "wav_name" and "mono"
def file_load(wav_name, mono=False):
    """
    load .wav file.

    wav_name : str
        target .wav file
    sampling_rate : int
        audio file sampling_rate
    mono : boolean
        When load a multi channels file and this param True, the returned data will be merged for mono data

    return : numpy.array( float )
    """
    try:
        # Load the .wav file using the librosa.load() function with parameters "wav_name", sr=None, and mono=mono
        return librosa.load(wav_name, sr=None, mono=mono)
    except:
        # Log an error message indicating that the file is broken or does not exist
        logger.error("file_broken or not exists!! : {}".format(wav_name))



# Define a function named "demux_wav" that takes in parameters "wav_name" and "channel"
def demux_wav(wav_name, channel=0):
    """
    demux .wav file.

    wav_name : str
        target .wav file
    channel : int
        target channel number

    return : numpy.array( float )
        demuxed mono data

    Enabled to read multiple sampling rates.

    Enabled even one channel.
    """
    try:
        # Call the function "file_load" with parameter "wav_name" and assign the returned values to "multi_channel_data" and "sr"
        multi_channel_data, sr = file_load(wav_name)
        # Check if the number of dimensions of "multi_channel_data" is less than or equal to 1
        if multi_channel_data.ndim <= 1:
            # Return "sr" and "multi_channel_data" as they are
            return sr, multi_channel_data

        # Return the specified channel of "multi_channel_data" as a numpy array
        return sr, numpy.array(multi_channel_data)[channel, :]

    except ValueError as msg:
        # Log a warning message with the value of "msg"
        logger.warning(f'{msg}')


########################################################################


########################################################################
# feature extractor
########################################################################
def file_to_vector_array(file_name,
                         n_mels=64,
                         frames=5,
                         n_fft=1024,
                         hop_length=512,
                         power=2.0):
    """
    convert file_name to a vector array.

    file_name : str
        target .wav file

    return : numpy.array( numpy.array( float ) )
        vector array
        * dataset.shape = (dataset_size, fearture_vector_length)
    """
    # 01 calculate the number of dimensions
    dims = n_mels * frames

    # 02 generate melspectrogram using librosa (**kwargs == param["librosa"])
    sr, y = demux_wav(file_name)
    mel_spectrogram = librosa.feature.melspectrogram(y=y,
                                                     sr=sr,
                                                     n_fft=n_fft,
                                                     hop_length=hop_length,
                                                     n_mels=n_mels,
                                                     power=power)

    # 03 convert melspectrogram to log mel energy
    log_mel_spectrogram = 20.0 / power * numpy.log10(mel_spectrogram + sys.float_info.epsilon)

    # 04 calculate total vector size
    vectorarray_size = len(log_mel_spectrogram[0, :]) - frames + 1

    # 05 skip too short clips
    if vectorarray_size < 1:
        return numpy.empty((0, dims), float)

    # 06 generate feature vectors by concatenating multi_frames
    vectorarray = numpy.zeros((vectorarray_size, dims), float)
    for t in range(frames):
        vectorarray[:, n_mels * t: n_mels * (t + 1)] = log_mel_spectrogram[:, t: t + vectorarray_size].T

    return vectorarray



# Define a function named "list_to_vector_array" that takes in parameters "file_list", "msg", "n_mels", "frames", "n_fft", "hop_length", and "power"
def list_to_vector_array(file_list,
                         msg="calc...",
                         n_mels=64,
                         frames=5,
                         n_fft=1024,
                         hop_length=512,
                         power=2.0):
    """
    convert the file_list to a vector array.
    file_to_vector_array() is iterated, and the output vector array is concatenated.

    file_list : list [ str ]
        .wav filename list of dataset
    msg : str ( default = "calc..." )
        description for tqdm.
        this parameter will be input into "desc" param at tqdm.

    return : numpy.array( numpy.array( float ) )
        training dataset (when generate the validation data, this function is not used.)
        * dataset.shape = (total_dataset_size, feature_vector_length)
    """
    # 01 calculate the number of dimensions
    dims = n_mels * frames

    # 02 loop of file_to_vectorarray
    # Iterate over the indices of the file_list using tqdm for progress visualization
    for idx in tqdm(range(len(file_list)), desc=msg):

        # Call the function file_to_vector_array with the current file from file_list and other parameters
        vector_array = file_to_vector_array(file_list[idx],
                                            n_mels=n_mels,
                                            frames=frames,
                                            n_fft=n_fft,
                                            hop_length=hop_length,
                                            power=power)

        # Check if it's the first iteration
        if idx == 0:
            # Create an empty numpy array with the shape of (vector_array.shape[0] * len(file_list), dims)
            dataset = numpy.zeros((vector_array.shape[0] * len(file_list), dims), float)

        # Assign the values of vector_array to the corresponding positions in the dataset array
        dataset[vector_array.shape[0] * idx: vector_array.shape[0] * (idx + 1), :] = vector_array

    # Return the dataset array
    return dataset



# Define a function named "dataset_generator" that takes in parameters "target_dir", "normal_dir_name", "abnormal_dir_name", and "ext"
def dataset_generator(target_dir,
                      normal_dir_name="normal",
                      abnormal_dir_name="abnormal",
                      ext="wav"):
    """
    target_dir : str
        base directory path of the dataset
    normal_dir_name : str (default="normal")
        directory name the normal data located in
    abnormal_dir_name : str (default="abnormal")
        directory name the abnormal data located in
    ext : str (default="wav")
        filename extension of audio files 

    return : 
        train_data : numpy.array( numpy.array( float ) )
            training dataset
            * dataset.shape = (total_dataset_size, feature_vector_length)
        train_files : list [ str ]
            file list for training
        train_labels : list [ boolean ] 
            label info. list for training
            * normal/abnormal = 0/1
        eval_files : list [ str ]
            file list for evaluation
        eval_labels : list [ boolean ] 
            label info. list for evaluation
            * normal/abnormal = 0/1
    """
    # Log the target directory
    logger.info("target_dir : {}".format(target_dir))

    # 01 normal list generate
    # Get a list of normal files by searching for files with the specified extension in the normal directory
    normal_files = sorted(glob.glob(
        os.path.abspath("{dir}/{normal_dir_name}/*.{ext}".format(dir=target_dir,
                                                                 normal_dir_name=normal_dir_name,
                                                                 ext=ext))))
    # Create an array of zeros with the length of the normal files list to represent normal labels
    normal_labels = numpy.zeros(len(normal_files))
    # If there are no normal files, log an exception
    if len(normal_files) == 0:
        logger.exception("no_wav_data!!")

    # 02 abnormal list generate
    # Get a list of abnormal files by searching for files with the specified extension in the abnormal directory
    abnormal_files = sorted(glob.glob(
        os.path.abspath("{dir}/{abnormal_dir_name}/*.{ext}".format(dir=target_dir,
                                                                   abnormal_dir_name=abnormal_dir_name,
                                                                   ext=ext))))
    # Create an array of ones with the length of the abnormal files list to represent abnormal labels
    abnormal_labels = numpy.ones(len(abnormal_files))
    # If there are no abnormal files, log an exception
    if len(abnormal_files) == 0:
        logger.exception("no_wav_data!!")

    # 03 separate train & eval
    # Separate the files and labels into training and evaluation sets
    train_files = normal_files[len(abnormal_files):]
    train_labels = normal_labels[len(abnormal_files):]
    eval_files = numpy.concatenate((normal_files[:len(abnormal_files)], abnormal_files), axis=0)
    eval_labels = numpy.concatenate((normal_labels[:len(abnormal_files)], abnormal_labels), axis=0)
    # Log the number of training and evaluation files
    logger.info("train_file num : {num}".format(num=len(train_files)))
    logger.info("eval_file  num : {num}".format(num=len(eval_files)))

    # Return the training files, training labels, evaluation files, and evaluation labels
    return train_files, train_labels, eval_files, eval_labels



########################################################################


########################################################################
# keras model
########################################################################

# Define a function named "keras_model" that takes in a parameter "inputDim"
def keras_model(inputDim):
    """
    define the keras model
    the model based on the simple dense auto encoder (64*64*8*64*64)
    """

    # Create an input layer with the specified input dimension
    inputLayer = Input(shape=(inputDim,))

    # Add a dense layer with 64 units and ReLU activation function
    h = Dense(64, activation="relu")(inputLayer)

    # Add another dense layer with 64 units and ReLU activation function
    h = Dense(64, activation="relu")(h)

    # Add a dense layer with 8 units and ReLU activation function
    h = Dense(8, activation="relu")(h)

    # Add another dense layer with 64 units and ReLU activation function
    h = Dense(64, activation="relu")(h)

    # Add another dense layer with 64 units and ReLU activation function
    h = Dense(64, activation="relu")(h)

    # Add a dense layer with the same number of units as the input dimension and no activation function
    h = Dense(inputDim, activation=None)(h)

    # Create a model with the input layer as input and the last dense layer as output
    return Model(inputs=inputLayer, outputs=h)


########################################################################


########################################################################
# main
########################################################################
if __name__ == "__main__":

    

    # Open the "baseline.yaml" file and load its contents into the "param" variable
    with open("baseline.yaml") as stream:
        param = yaml.safe_load(stream)

    # Create directories specified in the "param" dictionary if they don't already exist
    os.makedirs(param["pickle_directory"], exist_ok=True)
    os.makedirs(param["model_directory"], exist_ok=True)
    os.makedirs(param["result_directory"], exist_ok=True)

    # Create an instance of the "visualizer" class
    visualizer = visualizer()

    # Get a list of directories in the "base_directory" specified in the "param" dictionary
    dirs = sorted(glob.glob(os.path.abspath("{base}/*/*/*".format(base=param["base_directory"]))))

    # Define the path for the result file
    result_file = "{result}/{file_name}".format(result=param["result_directory"], file_name=param["result_file"])
    results = {}

    # Iterate over each directory in the "dirs" list
    for dir_idx, target_dir in enumerate(dirs):
        print("\n===========================")
        print("[{num}/{total}] {dirname}".format(dirname=target_dir, num=dir_idx + 1, total=len(dirs)))

        # Extract the database, machine type, and machine ID from the directory path
        db = os.path.split(os.path.split(os.path.split(target_dir)[0])[0])[1]
        # Extract the machine type and machine ID from the directory path
        machine_type = os.path.split(os.path.split(target_dir)[0])[1]
        machine_id = os.path.split(target_dir)[1]

        # Initialize variables for evaluation results and file paths
        evaluation_result = {}

        # Define the file path for the pickle file that stores the training data
        train_pickle = "{pickle}/train_{machine_type}_{machine_id}_{db}.pickle".format(pickle=param["pickle_directory"],
                                                                                       machine_type=machine_type,
                                                                                       machine_id=machine_id, db=db)

        # Define the file path for the pickle file that stores the evaluation files
        eval_files_pickle = "{pickle}/eval_files_{machine_type}_{machine_id}_{db}.pickle".format(
                                                                                       pickle=param["pickle_directory"],
                                                                                       machine_type=machine_type,
                                                                                       machine_id=machine_id,
                                                                                       db=db)

        # Define the file path for the pickle file that stores the evaluation labels
        eval_labels_pickle = "{pickle}/eval_labels_{machine_type}_{machine_id}_{db}.pickle".format(
                                                                                       pickle=param["pickle_directory"],
                                                                                       machine_type=machine_type,
                                                                                       machine_id=machine_id,
                                                                                       db=db)

        # Define the file path for the trained model
        model_file = "{model}/model_{machine_type}_{machine_id}_{db}.hdf5".format(model=param["model_directory"],
                                                                                  machine_type=machine_type,
                                                                                  machine_id=machine_id,
                                                                                  db=db)

        # Define the file path for the history image
        history_img = "{model}/history_{machine_type}_{machine_id}_{db}.png".format(model=param["model_directory"],
                                                                                    machine_type=machine_type,
                                                                                    machine_id=machine_id,
                                                                                    db=db)

        # Define the key for the evaluation result
        evaluation_result_key = "{machine_type}_{machine_id}_{db}".format(machine_type=machine_type,
                                                                          machine_id=machine_id,
                                                                          db=db)

        # Print a message indicating the start of dataset generation
        print("============== DATASET_GENERATOR ==============")

        # Check if the necessary pickle files exist, otherwise generate the dataset
        if os.path.exists(train_pickle) and os.path.exists(eval_files_pickle) and os.path.exists(eval_labels_pickle):
            # Load the pre-generated pickle files
            train_data = load_pickle(train_pickle)
            eval_files = load_pickle(eval_files_pickle)
            eval_labels = load_pickle(eval_labels_pickle)
        else:
            # Generate the dataset using the dataset_generator function
            train_files, train_labels, eval_files, eval_labels = dataset_generator(target_dir)

            # Convert the list of train_files to a vector array
            train_data = list_to_vector_array(train_files,
                                              msg="generate train_dataset",
                                              n_mels=param["feature"]["n_mels"],
                                              frames=param["feature"]["frames"],
                                              n_fft=param["feature"]["n_fft"],
                                              hop_length=param["feature"]["hop_length"],
                                              power=param["feature"]["power"])

            # Save the generated dataset as pickle files
            save_pickle(train_pickle, train_data)
            save_pickle(eval_files_pickle, eval_files)
            save_pickle(eval_labels_pickle, eval_labels)

        
        # Print a message indicating the start of model training
        print("============== MODEL TRAINING ==============")

        # Create a Keras model with the specified number of input features
        model = keras_model(param["feature"]["n_mels"] * param["feature"]["frames"])

        # Print a summary of the model architecture
        model.summary()

        # Check if a saved model file exists, and if so, load the weights into the model
        if os.path.exists(model_file):
            model.load_weights(model_file)
        else:
            # Compile the model with the specified parameters
            model.compile(**param["fit"]["compile"])

            # Train the model using the training data
            history = model.fit(train_data,
                                train_data,
                                epochs=param["fit"]["epochs"],
                                batch_size=param["fit"]["batch_size"],
                                shuffle=param["fit"]["shuffle"],
                                validation_split=param["fit"]["validation_split"],
                                verbose=param["fit"]["verbose"])

            # Plot and save the loss history of the model
            visualizer.loss_plot(history.history["loss"], history.history["val_loss"])
            visualizer.save_figure(history_img)

            # Save the trained model weights
            model.save_weights(model_file)

        # Print a message indicating the start of evaluation
        print("============== EVALUATION ==============")

        # Initialize an empty list for predicted labels and use the true labels from evaluation data
        y_pred = [0. for k in eval_labels]
        y_true = eval_labels

        # Iterate over each evaluation file and calculate the prediction error
        for num, file_name in tqdm(enumerate(eval_files), total=len(eval_files)):
            try:
                # Convert the evaluation file to a vector array
                data = file_to_vector_array(file_name,
                                            n_mels=param["feature"]["n_mels"],
                                            frames=param["feature"]["frames"],
                                            n_fft=param["feature"]["n_fft"],
                                            hop_length=param["feature"]["hop_length"],
                                            power=param["feature"]["power"])

                # Calculate the prediction error and store the mean error in the predicted labels list
                error = numpy.mean(numpy.square(data - model.predict(data)), axis=1)
                y_pred[num] = numpy.mean(error)
            except:
                logger.warning("File broken!!: {}".format(file_name))

        # Calculate the ROC AUC score using the true and predicted labels
        score = metrics.roc_auc_score(y_true, y_pred)

        # Log the AUC score
        logger.info("AUC : {}".format(score))

        # Store the AUC score in the evaluation result dictionary
        evaluation_result["AUC"] = float(score)

        # Store the evaluation result dictionary in the overall results dictionary
        results[evaluation_result_key] = evaluation_result
          
        # Print a separator line
        print("===========================")

        # Print a message indicating the end of evaluation
    print("\n===========================")
    # Log the path of the result file
    logger.info("all results -> {}".format(result_file))
    # Write the results dictionary to a YAML file
    with open(result_file, "w") as f:
        f.write(yaml.dump(results, default_flow_style=False))
    # Print a separator line
    print("===========================")

    

########################################################################
