from tensorflow.keras.datasets import mnist

def get_dataset():
    (trainX, trainY), (testX, testY) = mnist.load_data()
    return trainX, trainY, testX, testY