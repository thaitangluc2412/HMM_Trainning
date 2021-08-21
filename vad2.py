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
FILE_NAME="manhviet.wav"

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
    

if __name__ == '__main__':
    THESHOLD = count_THESHOLD()
    record(THESHOLD)
    
  
    

