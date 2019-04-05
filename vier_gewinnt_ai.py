import keras
import numpy as np

class vr_ai():
    def __init__(self, rows=6, cols=7, name='N/A'):
        self.net = self.net_init(rows, cols)
        self.rows = rows
        self.cols = cols
        self.name = name
        
    def net_init(self, rows, cols):
        model = keras.Sequential()
        
        #model.add(keras.layers.Reshape((1, rows, cols, 1)))
        model.add(keras.layers.Conv2D(64, (5, 5), padding='same', activation='relu', data_format='channels_last', input_shape=(rows, cols, 1)))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Conv2D(128, (5, 5), padding='same', activation='relu', data_format='channels_last'))
        model.add(keras.layers.Flatten())
        #model.add(keras.layers.Flatten(input_shape=(rows, cols)))
        #model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dense(rows*cols, activation='relu'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dense(10, activation='relu'))
        #model.add(keras.layers.Dense(32, activation='relu'))
        model.add(keras.layers.Dense(cols, activation='relu'))
        
        model.compile(loss='mse', optimizer='adam')
        
        return(model)
    
    def get_net(self):
        return(self.net)
    
    def get_move(self, state):
        return(np.argmax(self.get_q_values(state)))
    
    def get_q_values(self, state):
        net = self.get_net()
        o = net.predict(self.state_to_input(state)).flatten()
        #print(o)
        return(o)
    
    def train_step(self, next_step, target):
        net = self.get_net()
        
        net.fit(self.state_to_input(next_step), np.array([target]), verbose=0)
    
    def set_weights(self, arg):
        self.net.set_weights(arg)
    
    def get_weights(self):
        return(self.net.get_weights())
    
    def randomize_weights(self, ratio=0.25):
        f_weights = []
        for w in self.get_weights():
            s = w.size
            
            keys = []
            
            for i in range(s):
                if np.random.random() < ratio:
                    keys.append(tuple([np.random.choice(d) for d in w.shape]))
            
            keys = list(set(keys))
            
            for k in keys:
                val = abs(w[k])
                
                w[k] = 4 * val * np.random.random() - 2 * val
            
            f_weights.append(w)
        
        self.set_weights(f_weights)
    
    def state_to_input(self, state):
        return(np.array([state.reshape(self.rows, self.cols,1)]))
