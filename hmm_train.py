#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
class_names = ['dup', 'giu', 'nha', 'phai', 'trai']
states = [12, 19, 9, 12, 12]
# số trạng thái bằng số âm x3 - số âm = số âm thực sự + 1 khoảng lặng
length = 0
for d in class_names:
    length += len(os.listdir("data/" + d))
print(length)
# get_ipython().system('pip install librosa')
# get_ipython().system('pip install hmmlearn')


# In[2]:


import librosa
import numpy as np
import os
import math
from sklearn.cluster import KMeans
import hmmlearn.hmm

def get_mfcc(file_path):
    y, sr = librosa.load(file_path) # read .wav file 22050
    hop_length = math.floor(sr*0.010) # 10ms hop
    win_length = math.floor(sr*0.025) # 25ms frame
    # mfcc is 12 x T matrix
    mfcc = librosa.feature.mfcc(
        y, sr, n_mfcc=12, n_fft=1024,
        hop_length=hop_length, win_length=win_length)
    # substract mean from mfcc --> normalize mfcc
    mfcc = mfcc - np.mean(mfcc, axis=1).reshape((-1,1)) 
    # delta feature 1st order and 2nd order
    delta1 = librosa.feature.delta(mfcc, order=1)
    delta2 = librosa.feature.delta(mfcc, order=2)
    #bổ sung vào slide + báo cáo
    # X is 36 x T
    X = np.concatenate([mfcc, delta1, delta2], axis=0) # O^r
    # return T x 36 (transpose of X)
    return X.T # hmmlearn use T x N matrix


# In[3]:


# all_data = {}
# all_labels = {}
# for cname in class_names:
#     file_paths = [os.path.join("data", cname, i) for i in os.listdir(os.path.join('data', cname)) if i.endswith('.wav')]
#     data = [get_mfcc(file_path) for file_path in file_paths]
#     all_data[cname] = data
#     all_labels[cname] = [class_names.index(cname) for i in range(len(file_paths))]
#     print((all_data[cname]))


# In[4]:


# from sklearn.model_selection import train_test_split

# X = {'train': {}, 'test': {}}
# y = {'train': {}, 'test': {}}
# for cname in class_names:
#     x_train, x_test, _, y_test = train_test_split(
#         all_data[cname], all_labels[cname], 
#         test_size = 0.2, 
#         random_state=42
#     )
#     X['train'][cname] = x_train
#     X['test'][cname] = x_test
#     y['test'][cname] = y_test


# In[5]:


# print("Ten file", " Train", " Test")
# for cname in class_names:
#     print(" ", cname,"    ", len(X['train'][cname]), "   ",len(X['test'][cname]))


# In[6]:


# import hmmlearn.hmm as hmm

# model = {}
# for idx, cname in enumerate(class_names):
#     start_prob = np.full(states[idx], 0.0)
#     start_prob[0] = 1.0
#     trans_matrix = np.full((states[idx], states[idx]), 0.0)
#     p = 0.5
#     np.fill_diagonal(trans_matrix, p)
#     np.fill_diagonal(trans_matrix[0:, 1:], 1 - p)
#     trans_matrix[-1, -1] = 1.0
#     #trans matrix
#     model[cname] = hmm.GaussianHMM(
#         n_components=states[idx], 
#         verbose=True, 
#         n_iter=300, 
#         startprob_prior=start_prob, 
#         transmat_prior=trans_matrix,
#         params='stmc',
#         init_params='mc',
#         random_state=42
#     )
#     model[cname].fit(X=np.vstack(X['train'][cname]), lengths=[x.shape[0] for x in X['train'][cname]])
#     print(model[cname].__dict__)


# # In[7]:


# import pickle

# # save model
# for cname in class_names:
#     name = f'models_train\model_{cname}.pkl'
#     with open(name, 'wb') as file: 
#         pickle.dump(model[cname], file)


# # In[8]:


# import pickle, os
# import numpy as np

# from sklearn.metrics import classification_report


# # In[9]:


# y_true = []
# y_pred = []
# for cname in class_names:
#     for mfcc, target in zip(X['test'][cname], y['test'][cname]):
#         scores = [model[cname].score(mfcc) for cname in class_names]
#         pred = np.argmax(scores)
#         y_pred.append(pred)
#         y_true.append(target)
#     print((np.array(y_true) == np.array(y_pred)).sum()/len(y_true))
# print(y_true)
# print(y_pred)


# In[10]:


# report = classification_report(y_true, y_pred, target_names=class_names)
# print(report)


# In[11]:


#loadmodels
import pickle

model = {}
for key in class_names:
    name = f"models\model_{key}.pkl"
    with open(name, 'rb') as file:
        model[key] = pickle.load(file)


# In[12]:


print(model)


# In[13]:


# def my_confusion_matrix(y_true, y_pred):
#     N = np.unique(y_true).shape[0] # number of classes 
#     print(N)
#     cm = np.zeros((N, N))
#     print(cm)
#     for n in range(y_true.shape[0]):
#         cm[y_true[n], y_pred[n]] += 1
#     return cm 


# In[18]:


import os
import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
from utils.estnoise_ms import * 
import math
import pyaudio
import wave
from array import array

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024
RECORD_SECONDS = 1
FILE_NAME="RECORDING.wav"

def _calculate_frequencies(audio_data,sr):
        data_freq = np.fft.fftfreq(len(audio_data),1.0/sr)
        data_freq = data_freq[1:]
        return data_freq

def _calculate_amplitude(audio_data):
        data_ampl = np.abs(np.fft.rfft(audio_data,axis=0))
        data_ampl = data_ampl[1:]
        return data_ampl

def _calculate_energy(data):
        data_amplitude = _calculate_amplitude(data)
        data_energy = data_amplitude ** 2
        return data_energy

def _connect_energy_with_frequencies(data_freq, data_energy):
        energy_freq = {}
        for (i, freq) in enumerate(data_freq):
            if abs(freq) not in energy_freq:
                energy_freq[abs(freq)] = data_energy[i] * 2
        return energy_freq

def _calculate_normalized_energy(data,sr):
        data_freq = _calculate_frequencies(data,sr)
        data_energy = _calculate_energy(data)
        #data_energy = self._znormalize_energy(data_energy) #znorm brings worse results
        energy_freq = _connect_energy_with_frequencies(data_freq, data_energy)
        return energy_freq

def _sum_energy_in_band(energy_frequencies, start_band, end_band):
        sum_energy = 0
        for f in energy_frequencies.keys():
            if start_band<f<end_band:
               sum_energy += energy_frequencies[f]
        return sum_energy


def count_THESHOLD():
    audio = pyaudio.PyAudio()

    stream = audio.open(format=FORMAT, channels=CHANNELS,
                rate=RATE, input=True,
                frames_per_buffer=CHUNK)
    print ("recording...")        
    frames = array('h')
    
    for i in range(0, int(RATE/CHUNK*1)):
              data = stream.read(CHUNK)
              data_chunk = array('h', data)
              frames.extend(data_chunk)
             
    stream.stop_stream()
    stream.close()
    audio.terminate()

    print("Done")

    energy_freq = _calculate_normalized_energy(frames, RATE) 
    
    
    sum_full_energy = sum(energy_freq.values()) / 42 
    
    
    return sum_full_energy

def record(THESHOLD):
    
    audio = pyaudio.PyAudio()

    stream = audio.open(format=FORMAT, channels=CHANNELS,
                rate=RATE, input=True,
                frames_per_buffer=CHUNK)
    print ("recording...")        
    frames = array('h')
    i = 0

    while True:
        data = stream.read(CHUNK)
        data_chunk = array('h', data)
        frames.extend(data_chunk)
        i += 1
        if i == 4:
            energy_freq = _calculate_normalized_energy(frames, RATE) 
            sum_full_energy = sum(energy_freq.values()) / 4
            if sum_full_energy / THESHOLD > 1.2:
                print(sum_full_energy)
                continue
            else:
                i = 0
                frames = array('h')
        if i == 86:
            break    

    stream.stop_stream()
    stream.close()
    audio.terminate()
     
    wavfile=wave.open(FILE_NAME,'wb')
    wavfile.setnchannels(CHANNELS)
    wavfile.setsampwidth(audio.get_sample_size(FORMAT)) 
    wavfile.setframerate(RATE)
    wavfile.writeframes(frames)
    wavfile.close() 
    


# In[19]:


def predict():


        #Predict
        record_mfcc = get_mfcc("RECORDING.wav")
        scores = [model[cname].score(record_mfcc) for cname in class_names]
        predict_word = np.argmax(scores)
        print(scores)
        #class_names_vie = ['dup', 'giu', 'nha', 'phai', 'trai']
        print("Kết quả dự đoán: ", class_names[predict_word])   
        return class_names[predict_word]  
        
    


# In[24]:




# In[ ]:




