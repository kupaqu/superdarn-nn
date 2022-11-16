from DataLoader import DataLoader
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button

class LiveViewer:
    def __init__(self, datapath, gen_path, dis_path):
        self.loader = DataLoader(datapath=os.path.abspath(datapath))
        self.generator = tf.keras.models.load_model(os.path.abspath(gen_path))
        self.discriminator = tf.keras.models.load_model(os.path.abspath(dis_path))

        self.fig, self.ax = plt.subplots(6, 2)
        self.i = 0

        self.axprev = self.fig.add_axes([0.7, 0.05, 0.1, 0.075])
        self.axnext = self.fig.add_axes([0.81, 0.05, 0.1, 0.075])
        self.bnext = Button(self.axnext, 'Next')
        self.bnext.on_clicked(self.next)
        self.bprev = Button(self.axprev, 'Previous')
        self.bprev.on_clicked(self.prev)

        self.plot()

        plt.show()
    
    def plot(self):
        x, y = self.loader.xy(self.loader.sequence_for_learning[self.i])
        x, y = np.expand_dims(x[:,:,:,0,0], axis=0), np.expand_dims(y[:,:,:,0,0], axis=0)

        gen_pred = self.generator.predict(x)
        # dis_true_pred = self.discriminator.predict([x, y[:,:,:-1]])
        # dis_fake_pred = self.discriminator.predict([x, gen_pred])
        
        xy = np.concatenate([x, y], axis=2)
        xpred = np.concatenate([x[:,:,:,:-1], gen_pred], axis=2)
        

        for key in range(6):
            self.ax[key, 0].imshow(xy[0,:,:,key])
            self.ax[key, 1].imshow(xpred[0,:,:,key])
        
        plt.draw()
    
    def next(self, event):
        self.i += 1
        if self.i >= len(self.loader.sequence_for_learning):
            self.i = 0
        self.plot()

    def prev(self, event):
        self.i -= 1
        if self.i < 0:
            self.i = len(self.loader.sequence_for_learning)-1
        self.plot()

viewer = LiveViewer('./val', './best_generator.hdf5', './best_discriminator.hdf5')