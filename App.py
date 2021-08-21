class_names = ['cothe', 'khong', 'nguoi', 'toi', 'nhung']

import librosa
import numpy as np
import os
import math
from sklearn.cluster import KMeans
import hmmlearn.hmm
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import pyaudio
import wave
from base64 import b64decode


def get_mfcc(file_path):
    y, sr = librosa.load(file_path) # read .wav file
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
    # X is 36 x T
    X = np.concatenate([mfcc, delta1, delta2], axis=0) # O^r
    # return T x 36 (transpose of X)
    return X.T # hmmlearn use T x N matrix

#loadmodels
import pickle

model = {}
for key in class_names:
    name = f"models\model_{key}.pkl"
    with open(name, 'rb') as file:
        model[key] = pickle.load(file)
        
from tkinter import messagebox
import winsound

from pydub import AudioSegment

import ffmpeg

import tkinter as tk
from tkinter import *
from functools import partial

class Record():
    predict_word = ''
    
    def __init__(self, fileName):        
        self.fileName = fileName
        
    def changeFile():
        self.text.set(self.fileName)

#Thay đổi threshold dựa vào tạp âm, càng ồn thì threshold càng lớn
    def detect_leading_silence(self, sound, silence_threshold=-42.0, chunk_size=10):
        '''
        sound is a pydub.AudioSegment
        silence_threshold in dB
        chunk_size in ms

        iterate over chunks until you find the first one with sound
        '''
        trim_ms = 0 # ms

        assert chunk_size > 0 # to avoid infinite loop
        while sound[trim_ms:trim_ms+chunk_size].dBFS < silence_threshold and trim_ms < len(sound):
            trim_ms += chunk_size

        return trim_ms

    def chooseFile(self):

        Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
        self.fileName = askopenfilename() # show an "Open" dialog box and return the path to the selected file   
        self.changeFile
        print(self.fileName)

    def record(self):

        self.fileName = "record.wav"
        CHUNK = 1024
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 22050
        RECORD_SECONDS = 2
        WAVE_OUTPUT_FILENAME = "record.wav"

        p = pyaudio.PyAudio()

        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)

        frames = []

        for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK)
            frames.append(data)

        stream.stop_stream()
        stream.close()
        p.terminate()

        wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()

    def play(self):   
#         file = "record.wav"
        winsound.PlaySound(self.fileName, winsound.SND_FILENAME)

    def playtrimmed(self):    
        filename = 'trimmed.wav'
        winsound.PlaySound(filename, winsound.SND_FILENAME)

    def predict(self):
        #Trim silence
        sound = AudioSegment.from_file(self.fileName, format="wav")

        start_trim = self.detect_leading_silence(sound)
        end_trim = self.detect_leading_silence(sound.reverse())

        duration = len(sound)    

        trimmed_sound = sound[start_trim:duration-end_trim]    
        trimmed_sound.export("trimmed.wav", format="wav")

        #Predict
        record_mfcc = get_mfcc("trimmed.wav")
        scores = [model[cname].score(record_mfcc) for cname in class_names]
        self.predict_word = np.argmax(scores)
        messagebox.showinfo("result", class_names[self.predict_word])
        
    def UI(self): 
        
        window = tk.Tk()
        window.geometry("400x300")
        window.title("Application")

        frame0 = tk.Frame(master=window)
        frame0.pack()

        frame1 = tk.Frame(master=window)
        frame1.pack()

        frame2 = tk.Frame(master=window)
        frame2.pack()

        label = tk.Label(master=frame0, text="Speech recognition", fg="#4299e1", font='Helvetica 18 bold')
        label.pack(padx=5, pady=10)

        file_name = tk.Label(master=frame0, text="Record or open available sound.", fg="#2b6cb0", font='Helvetica 12 italic')
        file_name.pack(padx=5, pady=5)

        btn_open = tk.Button(master=frame0, width=28, height=1, text="Open sound file",fg="#4299e1", command=self.chooseFile)
        btn_open.pack(side=tk.LEFT, padx=20, pady=20)

        btn_record = tk.Button(master=frame1, width=13, height=2, text="Record", fg="#4299e1", command=self.record)
        btn_record.pack(side=tk.LEFT, padx=5, pady=5)

        btn_playback = tk.Button(master=frame1, width=13, height=2, text="Play", fg="#4299e1", command=self.play)
        btn_playback.pack(side=tk.LEFT, padx=5, pady=5)

        btn_predict = tk.Button(master=frame2, width=15, height=2, text="Predict", fg="#fff", bg="#4299e1", activebackground="#2b6cb0", command=self.predict)
        btn_predict.pack(side=tk.LEFT, padx=5, pady=15)

        window.mainloop()

Rec = Record('record.wav')

Rec.UI()