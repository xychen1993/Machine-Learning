import sys
import numpy as np
import matplotlib.pyplot as plt
import csv
import random
import tensorflow as tf
import tflearn

    
def main():
    n_classes = 4
    node_number = 10
    activation = 'tanh'
    n_epoch = 100
    batch_size = 20
    csv_file_name = 'spiral_data.csv'
    activations = ['linear','tanh','sigmoid','softmax','relu']
    node_numbers = np.array([pow(2,i) for i in range(1,7)],dtype = int)
    n_epochs = np.array([pow(2,i) for i in range(1,9)],dtype = int)
    batch_sizes = np.array([pow(2,i) for i in range(1,7)],dtype = int)
    
    position_train,classification_train = read_csv2(csv_file_name)
    
    class_array_train = to_categorical(classification_train,n_classes)
    data_linear = []
    data_non_linear = []
    
    for n_epoch in n_epochs:
        for batch_size in batch_sizes:
            linear = []
            linear.append(n_epoch)
            linear.append(batch_size)
            model_l = linear_classifier(position_train,class_array_train,n_classes, n_epoch, batch_size)
            linear.append(get_accuracy(position_train,classification_train,model_l))
            print "linear:",linear
            data_linear.append(linear)

            for activation in activations:
                for node_number in node_numbers:
                    temp = []
                    temp.append(node_number)
                    temp.append(activation)
                    temp.append(n_epoch)
                    temp.append(batch_size)

                    model = non_linear_classifier(position_train,class_array_train,n_classes, node_number, activation, n_epoch, batch_size)

                    temp.append(get_accuracy(position_train,classification_train,model))
                    print "non:",temp
                    data_non_linear.append(temp)
    
    write_csv(data_non_linear,'non_linear.csv')
    write_csv(data_linear,'linear.csv')


def linear_classifier(position_array, class_array, n_classes,n_epoch, batch_size):
    """
    Here you will implement a linear neural network that will classify the input data. The input data is
    an x, y coordinate (in 'position_array') and a classification for that x, y coordinate (in 'class_array'). The
    order of the data in 'position_array' corresponds with the order of the data in 'class_array', i.e., the ith element
    in 'position_array' is classified by the ith element in 'class_array'.

    Your neural network will have an input layer that has two input nodes (an x coordinate and y coordinate)
    and an output layer that has four nodes (one for each class) with a softmax activation.

    :param position_array: a 2D np array of size [n_examples, 2] that contains an x,y position for each point
    :param class_array: a 1D np array of size [n_examples]
    :param n_classes: an integer that is the number of classes your data has
    """

    # linear classifier

    with tf.Graph().as_default():

        net = tflearn.input_data(shape=[None, n_classes])
        net = tflearn.fully_connected(net, 4, activation='softmax')
        net = tflearn.regression(net, loss='categorical_crossentropy')
        model = tflearn.DNN(net)
        model.fit(position_array, class_array, n_epoch=n_epoch, batch_size=batch_size, show_metric=True, snapshot_epoch = False)
        return model

def non_linear_classifier(position_array, class_array, n_classes, node_number, activation, n_epoch, batch_size):
    """
    Here you will implement a non-linear neural network that will classify the input data. The input data is
    an x, y coordinate (in 'position_array') and a classification for that x, y coordinate (in 'class_array'). The
    order of the data in 'position_array' corresponds with the order of the data in 'class_array', i.e., the ith element
    in 'position_array' is classified by the ith element in 'class_array'.

    Your neural network should have three layers total. An input layer and two fully connected layers
    (meaning that the middle layer is a hidden layer). The second fully connected layer is the output
    layer (so it should have 4 nodes and a softmax activation function). You get to decide how many
    nodes the middle layer has and the activation function that it uses.

    :param position_array: a 2D np array of size [n_examples, 2] that contains an x,y position for each point
    :param class_array: a 1D np array of size [n_examples]
    :param n_classes: an integer that is the number of classes your data has
    """
    with tf.Graph().as_default():

        net = tflearn.input_data(shape=[None, n_classes])
        net = tflearn.fully_connected(net, node_number, activation=activation)
        net = tflearn.fully_connected(net, node_number, activation=activation)
        net = tflearn.regression(net, optimizer = 'sgd', learning_rate = 1., loss='categorical_crossentropy')
        model = tflearn.DNN(net)
        model.fit(position_array, class_array, n_epoch=n_epoch, batch_size=batch_size, show_metric=True,snapshot_epoch = False)

        return model

def write_csv(data,path_to_file):
    csvfile = file(path_to_file, 'a+')
    writer = csv.writer(csvfile)
    writer.writerows(data)
    csvfile.close()

def read_csv(path_to_file):
    """
    Reads the csv file to input
    :param path_to_file: path to the csv file
    :return: a numpy array of positions, and a numpy array of classifications
    """
    r = np.array([i for i in range(400)])
    random.shuffle(r)
    r = r[:40]
    


    position_test = []
    classification_test = []
    position_train = []
    classification_train = []

    with open(path_to_file, 'r') as csv_file:
        reader = csv.reader(csv_file)
        next(reader, None) # skip the header

        for i,row in enumerate(reader):
            if i in r:
                position_test.append(np.array([float(row[0]), float(row[1])]))
                classification_test.append(float(row[2]))
            else:
                position_train.append(np.array([float(row[0]), float(row[1])]))
                classification_train.append(float(row[2]))

    return np.array(position_train), np.array(classification_train, dtype='uint8'),np.array(position_test), np.array(classification_test, dtype='uint8')

def read_csv2(path_to_file):
    """
    Reads the csv file to input
    :param path_to_file: path to the csv file
    :return: a numpy array of positions, and a numpy array of classifications
    """
    position_train = []
    classification_train = []
    with open(path_to_file, 'r') as csv_file:
        reader = csv.reader(csv_file)
        next(reader, None) # skip the header

        for i,row in enumerate(reader):
            position_train.append(np.array([float(row[0]), float(row[1])]))
            classification_train.append(float(row[2]))

    return np.array(position_train), np.array(classification_train, dtype='uint8')


def to_categorical(y, nb_classes):
    """ to_categorical.

        Convert class vector (integers from 0 to nb_classes)
        to binary class matrix, for use with categorical_crossentropy.

        Arguments:
            y: `array`. Class vector to convert.
            nb_classes: `int`. Total number of classes.
    """
    y = np.asarray(y, dtype='int32')
    if not nb_classes:
        nb_classes = np.max(y)+1
    Y = np.zeros((len(y), nb_classes))
    for i in range(len(y)):
        Y[i, y[i]] = 1.
    return Y

def get_accuracy(position_array, class_array, model):
    """
    Gets the accuracy of your model
    :param position_array: a 2D np array of size [n_examples, 2] that contains an x,y position for each point
    :param class_array: a 1D np array of size [n_examples]
    :param model: a tflearn model
    :return: a float in the range [0.0, 1.0]
    """
    return np.mean(class_array == np.argmax(model.predict(position_array), axis=1))

if __name__ == '__main__':
    main()