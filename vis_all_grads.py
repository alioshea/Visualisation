import numpy as np
from scipy.signal import butter, lfilter, freqz, decimate
from generate_eeg import get_eeg
import matplotlib.pyplot as plt
import os
import pdb

from keras import backend as K
from keras.layers.core import Activation
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import GlobalAveragePooling1D, AveragePooling1D
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.optimizers import SGD
from model import model

# dimensions of the generated pictures for each filter.
sample_length = 256  # 512

# Simple MAF for smoothing
window_size = 3


def movingaverage(data, window_size):
    data = data  # [:,0]
    window = np.ones(int(window_size)) / float(window_size)
    return np.convolve(data, window, "same")


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


# filter options
order = 6
fs_ft = 32
cutoff = 12.8

# step size for gradient ascent
step = 1

nb_filters = 2

babies = ['01', '02', '03', '04', '05', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19']

architecture = []
for baby in range(18):
    architecture.append(model(babies[baby], sample_length))


def normalize(x, norm_param=1e-9):
    # utility function to normalize a tensor by its l2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + norm_param)  # -5?

lam = 0.0001
best_probability = 0
input_signal = get_eeg()
while best_probability < 0.93:  # The average probability of that class

    average_grads = 0
    for network in range(18):
        input_layer = architecture[network].input
        output_layer = architecture[network].layers[-3].output

        output = K.mean(output_layer[:,:,:][0],axis=0)
        loss = 1/(1+K.exp(output[0])/K.exp(output[1])) - lam*K.dot(input_layer[0,:,0], K.transpose(input_layer[0,:,0]))
        grads = K.gradients(loss, input_layer) 

        iterate = K.function([input_layer, K.learning_phase()], [loss, grads, output])

        loss_value, grads_value, output_value = iterate([input_signal, 0])

        average_grads += grads_value*(1-1/(1+np.exp(output_value[0])/np.exp(output_value[1])))

    input_signal += average_grads * step
    current_losses = 0
    #pdb.set_trace()
    for network in range(18):
        model_loss = architecture[network].predict(input_signal)
        current_losses += model_loss[0, 1]
        print model_loss[0, 1] ,
    #print('\n')
    probability = current_losses / 18.0

    if probability > best_probability:
        best_probability = probability
        best_signal = input_signal
        print(best_probability)
        #plt.plot(input_signal[0])
        #plt.show()

np.save('SeizureOptimisedfor18NetworksV4.npy',input_signal[0])
plt.plot(input_signal[0])
#plt.show()

best_probability = 0
input_signal = get_eeg()
while best_probability < 0.98:  # The average probability of that class

    average_grads = 0
    for network in range(18):
        input_layer = architecture[network].input
        output_layer = architecture[network].layers[-3].output

        output = K.mean(output_layer[:,:,:][0],axis=0)
        loss = 1/(1+K.exp(output[1])/K.exp(output[0]))
        grads = K.gradients(loss, input_layer)

        iterate = K.function([input_layer, K.learning_phase()], [loss, grads, output])

        loss_value, grads_value, output_value = iterate([input_signal, 0])

        average_grads += grads_value*(1-1/(1+np.exp(output_value[0])/np.exp(output_value[1])))

    input_signal += average_grads * step
    current_losses = 0
    for network in range(18):
        model_loss = architecture[network].predict(input_signal)
        current_losses += model_loss[0, 0]
        print model_loss[0, 0] ,
    #plt.plot(input_signal[0])
    #plt.show()
    #print('\n')
    probability = current_losses / 18.0

    if probability > best_probability:
        print(model_loss)
        best_probability = probability
        best_signal = input_signal
        print(best_probability)

plt.figure()
plt.plot(input_signal[0])
np.save('BackgroundOptimisedfor18NetworksV4.npy',input_signal[0])
plt.show()
