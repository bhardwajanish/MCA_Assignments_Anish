# Sources
#https://stackoverflow.com/questions/54903873/calculating-spectrogram-of-wav-files-in-python
#https://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html
#https://www.kaggle.com/ybonde/log-spectrogram-and-mfcc-filter-bank-example
#https://github.com/numpy/numpy/blob/master/numpy/fft/_pocketfft.py

import numpy as np
import scipy.io.wavfile
import matplotlib.pyplot as plt

def signal_framing(signal, size = 0.025, stride = 0.01):
	length = int(round(sample_rate*size))
	step = int(round(sample_rate*stride))
	signallength = len(signal)
	framecount = int(round((signallength-length)/step))
	paddedlength = framecount *step +length
	paddedsignal = np.append(signal, np.zeros(paddedlength-signallength))
	indices = np.tile(np.arange(0, length), (framecount, 1)) + np.tile(np.arange(0, framecount * step, step), (length, 1)).T
	frames = paddedsignal[indices]
	windowedframes = frames*np.hamming(length)
	return windowedframes

# def ffttry(a, n, axis=-1):
# 	a = asarray(a)
#     inv_norm = 1
#     output = _raw_fft(a, n, axis, False, True, inv_norm)
#     return output

def FFTPow(frames, Npoints=512):
	fft = np.absolute(np.fft.fft(frames, int(Npoints/2)+1)) 
	powspectrum = ((fft ** 2)/Npoints) 
	return powspectrum

def MelScale(sample_rate, filt = 40):
	#lowrange = 0
	filters = filt + 2
	peak = 2595 * np.log10(1 + (sample_rate / 2) / 700)
	melscale = np.linspace(0, peak, filters)
	meltohz = (700 * (10**(melscale / 2595) - 1))
	return meltohz


def Filters(scale, sample_rate, trifilter = 40, Npoints = 512):
	filters = np.zeros((trifilter, int(np.floor(Npoints / 2)+1)))
	for i in range(trifilter):
	    left = int(scale[i])
	    center = int(scale[i+1]) 
	    right = int(scale[i+2])
	    for j in range(left, center):
	        filters[i, j] = (j - scale[i]) / (scale[i+1] - scale[i])
	    for j in range(center, right):
	        filters[i, j] = (scale[i+2] - j) / (scale[i+2] - scale[i])
	return filters

def  Spectrogram(powerspectrum, filters):
	spectro = np.dot(powerspectrum, filters.T)
	spectrogram = 20 * np.log10(spectro) 
	return spectrogram

fsize = 0.025
fstride = 0.01

sample_rate, signal = scipy.io.wavfile.read('test1.wav')
frames = signal_framing(signal, fsize, fstride)

Npoints = 512
ps = FFTPow(frames, Npoints)

trifilt = 40
meltohz = MelScale(sample_rate, trifilt)

scalehz = np.ceil((Npoints + 1) * meltohz / sample_rate)

filters = Filters(scalehz, sample_rate, trifilt, Npoints)

spectrogram = Spectrogram(ps, filters)
# print(spectrogram)

plt.figure()
plt.imshow(spectrogram)
plt.savefig('spectrogram.png')