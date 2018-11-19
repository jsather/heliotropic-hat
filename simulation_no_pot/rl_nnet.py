""" rl_nnet.py is where the Q network lives. It also contains the experience
    replay class and methods for saving and loading models.
    
    Author: Jonathon Sather
    Last updated: 1/03/2017 
"""

from collections import deque
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers.normalization import BatchNormalization
from keras.optimizers import RMSprop
from keras.optimizers import adam
from keras.optimizers import Nadam
from keras.regularizers import l2
import numpy as np 
import pdb

# Make random seed for predictability
np.random.seed(800800)

def load_model(location):
    """ Loads model from location and returns model. """

    return load_model(location)

class queue:
    """ Queue object for storing past actions. Contains option to initialize
        to filled with zeros.
    """

    def __init__(self, size, zeros=True):
        """ Initialize queue. Initialized to zeros by default."""
        
        if zeros:
            elements = [0] * size
        else:
            elements = [None] * size

        self.elements = deque(elements, maxlen=size)
        self.size = size

    def get_contents(self):
        """ Returns contents of queue in order. """
        contents = []

        for element in self.elements:
            contents.append(element)
        
        return contents

    def add(self, new):
        """ Adds an element to the front of the queue. """

        self.elements.appendleft(new)

    def re_init(self, zeros=True):
        """ Reinitializes queue. """

        if zeros:
            elements = [0] * self.size
        else:
            elements = [None] * size

        self.elements = deque(elements, maxlen=self.size)

class experienceReplay:
    """ Experience replay object used in training nnet. Stores size elements in
        buffer with methods to add and remove elements.
    """

    def __init__(self, size):
        """ Initialize experience replay buffer. """

        self.buf = []
        self.size = size # Max elements in buffer.
    
    def remove(self, number, random=True, pop=True, numReward=None, reward=1):
        """ Remove number elements from buffer. Random selection by
            default. Elements popped from buffer by default.
        """
        removed = []
        rewards = 0
        
        if numReward == None:
            for element in range(number):
                if random:
                    try:
                        remove = np.random.randint(0, len(self.buf))
                    except ValueError:
                        print("Nothing to remove!\nBuffer length: " +
                              str(len(self.buf)))
                        break
                else:
                    remove = len(self.buf) - (number - element)

                if pop:
                    try:
                        removed.append(self.buf.pop(remove))
                    except ValueError:
                        print("Can't remove " + str(remove) + " element.\n"
                              + "Buffer size: " + str(len(self.buf)))
                else:
                    try:
                        removed.append(self.buf[remove])
                    except ValueError:
                        print("Can't remove " + str(remove) + " element.\n"
                              + "Buffer size: " + str(len(self.buf)))
        else: 
            # Create batch with num. rewards. Assume random=True and pop=False 
            element = 0
            while element < numReward: # Get rewards to meet quota
                remove = np.random.randint(0, len(self.buf))
                if self.buf[remove][2] == reward:
                    removed.append(self.buf[remove])
                    element += 1
                else:
                    pass
            while element < number: # Get non rewards
                remove = np.random.randint(0, len(self.buf))
                if self.buf[remove][2] == reward:
                    pass
                else:
                    removed.append(self.buf[remove])
                    element += 1

        return removed

    def add(self, elements):
        """ Add elements to the buffer. Remove elements if needed. """

        if len(elements) > self.size:
            print("Cannot add " + str(len(elements)) + " elements to a size "
                  + str((self.size)) + " buffer.")
            return 0

        self.remove(max(0, len(self.buf) + len(elements) - self.size))
        self.buf.extend(elements)

        return 1

    def add_tuple(self, element):
        """ Add one tuple element to buffer. Remove element if needed. """

        self.remove(max(0, len(self.buf) + 1 - self.size), random=False, pop=True)
        self.buf.append(element)

        return 1

    def is_full(self):
        """ Check to see if buffer full. Returns 1 if full. 0 Otherwise. """

        if len(self.buf) == self.size:
            return 1
        else:
            return 0

    def has_at_least(self, num):
        """ Check to see if buffer has at least num elements. Returns 1 if has
            at least num. 0 otherwise.
        """

        if len(self.buf) >= num:
            return 1
        else:
            return 0

class QNetwork:
    """ Holds sequential model for Q learning. """

    def __init__(self, input_dim=21, output_dim=2, lr=0.00001, l2_val=0.01):
        """ Initialize neural network model. """

        # Create sequential neural network model
        model = Sequential()

        model.add(BatchNormalization(input_shape=(input_dim,)))
        model.add(Dense(50, input_dim=input_dim, W_regularizer=l2(l2_val),
                        init='normal', activation='relu'))

        model.add(BatchNormalization())
        model.add(Dense(10,  init='normal', W_regularizer=l2(l2_val),
                        activation='relu'))

        model.add(BatchNormalization())
        model.add(Dense(10,  init='normal', W_regularizer=l2(l2_val),
                        activation='relu'))

        model.add(BatchNormalization())
        model.add(Dense(10,  init='normal', W_regularizer=l2(l2_val), 
                        activation='relu'))
        
        model.add(BatchNormalization())
        model.add(Dense(output_dim, init='normal', W_regularizer=l2(l2_val),
                        activation='linear'))

        adam_opt = adam(lr=lr)
        model.compile(loss='mean_squared_error', optimizer=adam_opt)

        self.model = model

    def train_batch(self, training_data_x, training_data_y):
        """ Train neural network with batch. """
        
        return self.model.train_on_batch(training_data_x, training_data_y)

    def run_forward(self, training_data_x):
        """ Predict output of neural network."""

        return self.model.predict_on_batch(training_data_x)

    def save_model(self, location):
        """ Saves model at "location". """

        self.model.save(location)

    def new_model(self, new_model):
        """ Changes self.model to new_model. """

        self.model = new_model

if __name__ == '__main__':
    # Test code
    replay = experienceReplay(100)

    e1 = [np.empty((10,10))] * 100
    e2 = [np.empty((10,10))] * 101
    e3 = [np.empty((10,10))] * 500
    e4 = [np.empty((10,1))] * 10

    f = ['cat', 'cat1', 'frogdog']
    f2 = ['pooo', 'catpoo']
    f3 = ['jon'] * 99

   