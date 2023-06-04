from data_loader import get_dataset
from model import DNNClassifier
import numpy as np
from tensorflow.keras.utils import to_categorical


def fix_x_data(x_list):
    '''
    Expects x_list to contain two lists,  one the test andone  the training data
    '''
    for i, x_data in enumerate(x_list):
        x_list[i] = np.array([element.flatten() for element in x_data])
    return x_list[0], x_list[1]

def fix_y_data(y_list):
    '''
    Expects y_list to contain two lists,  one the test andone  the training data
    '''
    for i, y_data in enumerate(y_list):
        y_list[i] = to_categorical(y_data)
    return y_list[0], y_list[1]

if __name__ == '__main__':

    trainX, trainY, testX, testY = get_dataset()
    trainX, testX = fix_x_data([trainX, testX])
    output_dim = len(np.unique(trainY.flatten()))
    trainY, testY = fix_y_data([trainY, testY])

    classifier = DNNClassifier(input_dim=trainX.shape, 
                               output_dim=output_dim, 
                               n_layers=1, 
                               n_nodes=128)
    classifier.summary()
    classifier.compile()
    classifier.fit(trainX, trainY, epochs=40)
    classifier.evaluate(testX, testY)
