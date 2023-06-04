from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, InputLayer, Softmax
from tensorflow.keras.losses import CategoricalCrossentropy


class DNNClassifier:

    def  __init__(self, input_dim, output_dim, n_layers, n_nodes):
        '''
        Creates a simple DNN classifier with inputs:
        * input_dim: the dimension of the input, must be one-dimensional (n)
        * output_dim: the dimension of the output, must have n dimensions, according to the number of elements
            we want to make predictions for
        * n_layers: the number of layers in the classifier
        * n_nodes: the number of nodes in each of the layers in the model
        '''
        self.model = Sequential()
        self.model.add(InputLayer(input_dim[1]))
        for n in range(n_layers):
            self.model.add(Dense(n_nodes, activation='relu'))
        self.model.add(Dense(output_dim, activation='linear'))
        self.model.add(Softmax())
    
    def summary(self):
        print(self.model.summary())
    
    def compile(self):
        self.model.compile(loss=CategoricalCrossentropy(), optimizer='adam', metrics=['accuracy'])

    def fit(self, X, y, epochs=100):
        self.model.fit(X, y, epochs=epochs)

    def evaluate(self, X, y):
        self.model.evaluate(X, y)