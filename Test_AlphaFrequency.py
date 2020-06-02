# -*- coding: utf-8 -*-
"""
Created on Fri May  8 09:30:49 2020

@author: Mattek10b

This script perform the alpha wave frequency analysis on the EEG data sets of
one test subject at the time, available data sets are:
    - S1_CClean.mat
    - S1_OClean.mat
    - S3_CClean.mat
    - S3_OClean.mat
    - S4_CClean.mat
    - S4_OClean.mat
This script need the module:
    - Main_Algorithm
    - Data_EEG
    - ICA_Fast
The script can be run directly. If one need results from another data set,
change data_name_C or data_name_O to look at another test subject.
"""
from Main_Algorithm import Main_Algorithm
import Data_EEG
import ICA_Fast
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import butter, lfilter
from scipy.signal import freqz


np.random.seed(1234)

# =============================================================================
# Import EEG data file
# =============================================================================
data_name_C = 'S1_CClean.mat'                # Closed eyes - Subject 1
data_name_O = 'S1_OClean.mat'                # Open eyes - Subject 1
data_file_C = 'data/' + data_name_C          # file path
data_file_O = 'data/' + data_name_O          # file path

segment_time = 1                             # length of segments i seconds

# =============================================================================
# ICA
# =============================================================================
" ICA - Closed Eyes Dataset "
Y_ica_C, M_ica_C, L_ica_C, n_seg_ica_C = Data_EEG._import(data_file_C, segment_time, request='none')
X_ica_C, k_C = ICA_Fast.X_ica(data_name_C, Y_ica_C, M_ica_C)

" ICA - Open Eyes Dataset "
Y_ica_O, M_ica_O, L_ica_O, n_seg_ica_O = Data_EEG._import(data_file_O, segment_time, request='none')
X_ica_O, k_O = ICA_Fast.X_ica(data_name_O, Y_ica_O, M_ica_O)

# =============================================================================
# Main Algorithm with random A
# =============================================================================
" Import Segmented Dataset "
#request='remove 1/2' # remove sensors and the same sources from dataset
#request='remove 1/3' # remove sensors and the same sources from dataset
request = 'none'

" Main - Closed Eyes Dataset "
Y_C, M_C, L_C, n_seg_C = Data_EEG._import(data_file_C, segment_time, request=request)
A_C, X_C = Main_Algorithm(Y_C, M_C, k_C, L_C, data = 'EEG', fix = True)

" Main - Open Eyes Dataset "
Y_O, M_O, L_O, n_seg_O = Data_EEG._import(data_file_O, segment_time, request=request)
A_O, X_O = Main_Algorithm(Y_O, M_O, k_O, L_O, data = 'EEG', fix = True)
# =============================================================================
# DFT
# =============================================================================
segment = 35
row = 10

Y_C_signal = Y_C[segment][row]                  # one measurement signal
Y_O_signal = Y_O[segment][row]                  # one measurement signal
X_C_signal = X_C[segment][row]                  # one recovered source signal
X_O_signal = X_O[segment][row]                  # one recovered source signal

X_C_matrix = X_C[segment]
X_O_matrix = X_O[segment]
Y_C_matrix = Y_C[segment]
Y_O_matrix = Y_O[segment]

X_C_time = np.linspace(0, 1, len(X_C_signal)) # time signal (0ne second) for source signal
X_O_time = np.linspace(0, 1, len(X_O_signal)) # time signal (0ne second) for measurment signal
Y_C_time = np.linspace(0, 1, len(Y_C_signal)) # time signal (0ne second) for source signal
Y_O_time = np.linspace(0, 1, len(Y_O_signal)) # time signal (0ne second) for measurment signal

def DFT(signal):
    fft = np.fft.rfft(signal)    # FFT of signal
    fft_power = np.abs(fft)      # |FFT|
    return fft, fft_power
     
X_C_fft, X_C_power = DFT(X_C_signal)   # FFT of source signal
X_O_fft, X_O_power = DFT(X_O_signal)   # FFT of source signal

Y_C_fft, Y_C_power = DFT(Y_C_signal)   # FFT of measurement signal
Y_O_fft, Y_O_power = DFT(Y_O_signal)   # FFT of measurement signal

def DFT_matrix(matrix):
    fft = np.fft.rfft2(matrix)    # FFT of matrix
    fft_power = np.abs(fft)       # |FFT|
    return fft, fft_power

X_C_fft_matrix, X_C_power_matrix = DFT_matrix(X_C_matrix)   # FFT of source matrix
X_O_fft_matrix, X_O_power_matrix = DFT_matrix(X_O_matrix)   # FFT of source matrix

Y_C_fft_matrix, Y_C_power_matrix = DFT_matrix(Y_C_matrix)   # FFT of source matrix
Y_O_fft_matrix, Y_O_power_matrix = DFT_matrix(Y_O_matrix)   # FFT of source matrix

# =============================================================================
# Butterworth Bandpass filter
# =============================================================================
lowcut = 8     # low cut off frequency (Hz)
highcut = 13   # high cut off frequency (Hz)
fs = 512       # sample frequency
order = 5      # ordre of Butterworth filter

def butter_bandpass(lowcut, highcut, fs, order):
    nyq = 0.5 * fs      # nyquist
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='bandpass') # coefficients of transfer function
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)   # filter the signal with FIR filter
    return y

" Frequency Response "
b, a = butter_bandpass(lowcut, highcut, fs, order) # coefficients of transfer function
w, h = freqz(b, a, worN=2000)                      # frequency and frequency response
hz = (fs * 0.5 / np.pi) * w                        # frequncies

# =============================================================================
# Filtering
# =============================================================================
" X signal and Y signal Filtering "
def filtering(signal, lowcut, highcut, fs, order=5):
    filt = butter_bandpass_filter(signal, lowcut, highcut, fs, order)
    fft_filt = np.fft.rfft(filt)
    fft_power_filt = np.abs(fft_filt)
    return filt, fft_filt, fft_power_filt

X_C_filter, X_C_fft_filter, X_C_power_filter = filtering(X_C_signal, lowcut, highcut, fs, order=5)
X_O_filter, X_O_fft_filter, X_O_power_filter = filtering(X_O_signal, lowcut, highcut, fs, order=5)

Y_C_filter, Y_C_fft_filter, Y_C_power_filter = filtering(Y_C_signal, lowcut, highcut, fs, order=5)
Y_O_filter, Y_O_fft_filter, Y_O_power_filter = filtering(Y_O_signal, lowcut, highcut, fs, order=5)

" X matrix and Y matrix Filtering "
def filtering_matrix(matrix, lowcut, highcut, fs, order=5):
    filter_matrix = []
    for i in range(len(matrix)):
        filter_matrix.append(butter_bandpass_filter(matrix[i], lowcut, highcut, fs, order))
    fft_filter_matrix = np.fft.rfft2(filter_matrix)
    fft_power_filter_matrix = np.abs(fft_filter_matrix)
    return filter_matrix, fft_filter_matrix, fft_power_filter_matrix

X_C_filter_matrix, X_C_fft_filter_matrix, X_C_power_filter_matrix = filtering_matrix(X_C_matrix, lowcut, highcut, fs, order=5)
X_O_filter_matrix, X_O_fft_filter_matrix, X_O_power_filter_matrix = filtering_matrix(X_O_matrix, lowcut, highcut, fs, order=5)

Y_C_filter_matrix, Y_C_fft_filter_matrix, Y_C_power_filter_matrix = filtering_matrix(Y_C_matrix, lowcut, highcut, fs, order=5)
Y_O_filter_matrix, Y_O_fft_filter_matrix, Y_O_power_filter_matrix = filtering_matrix(Y_O_matrix, lowcut, highcut, fs, order=5)

# =============================================================================
# Average Differences
# =============================================================================
print('Average difference (Signal) between Y Closed and Y Open: ', abs(np.average(Y_C_power_filter/Y_O_power_filter)))
print('Average difference (Signal) between X Closed and X Open: ', abs(np.average(X_C_power_filter/X_O_power_filter)))

diff_sum_Y = []
diff_sum_X = []
for i in range(len(Y_C_filter_matrix)):    
    diff_sum_Y.append(abs(np.average(Y_C_fft_filter_matrix[i]/Y_O_fft_filter_matrix[i])))
for i in range(len(X_O_filter_matrix)):    
    diff_sum_X.append(abs(np.average(X_C_fft_filter_matrix[i]/X_O_fft_filter_matrix[i])))

print('Average difference (Matrix) between Y Closed and Y Open: ', np.average(diff_sum_Y))
print('Average difference (Matrix) between X Closed and X Open: ', np.average(diff_sum_X))


# =============================================================================
# DFT of all time segments
# =============================================================================
X_sum_C = np.zeros(100)
Y_sum_C = np.zeros(100)
X_sum_O = np.zeros(100)
Y_sum_O = np.zeros(100)
average_diff_X = np.zeros(100)
average_diff_Y = np.zeros(100) 
average_diff2_X = np.zeros(100)
average_diff2_Y = np.zeros(100)
for seg in range(100):
    X_C_fft_matrix, X_C_power_matrix = DFT_matrix(X_C[seg])      
    Y_C_fft_matrix, Y_C_power_matrix = DFT_matrix(Y_C[seg])   
    X_C_filter_matrix, X_C_fft_filter_matrix, X_C_power_filter_matrix = filtering_matrix(X_C[seg], lowcut, highcut, fs, order=5)
    Y_C_filter_matrix, Y_C_fft_filter_matrix, Y_C_power_filter_matrix = filtering_matrix(Y_C[seg], lowcut, highcut, fs, order=5)

    X_sum_C[seg] = np.average(sum(X_C_power_filter_matrix))
    Y_sum_C[seg] = np.average(sum(Y_C_power_filter_matrix))
    if seg == segment:
        X_C_power_filter_matrixSEG = np.array(X_C_power_filter_matrix, copy=True)
        Y_C_power_filter_matrixSEG = np.array(Y_C_power_filter_matrix, copy=True)
   

    X_O_fft_matrix, X_O_power_matrix = DFT_matrix(X_O[seg])
    Y_O_fft_matrix, Y_O_power_matrix = DFT_matrix(Y_O[seg])
    X_O_filter_matrix, X_O_fft_filter_matrix, X_O_power_filter_matrix = filtering_matrix(X_O[seg], lowcut, highcut, fs, order=5) 
    Y_O_filter_matrix, Y_O_fft_filter_matrix, Y_O_power_filter_matrix = filtering_matrix(Y_O[seg], lowcut, highcut, fs, order=5)
    
    X_sum_O[seg] = np.average(sum(X_O_power_filter_matrix))
    Y_sum_O[seg] = np.average(sum(Y_O_power_filter_matrix))

    if seg == segment:
        X_O_power_filter_matrixSEG = np.array(X_O_power_filter_matrix, copy=True)
        Y_O_power_filter_matrixSEG = np.array(Y_O_power_filter_matrix, copy=True)

    average_diff_X[seg] = np.average(sum(X_C_power_filter_matrix))/np.average(sum(X_O_power_filter_matrix))
    average_diff_Y[seg] = np.average(sum(Y_C_power_filter_matrix))/np.average(sum(Y_O_power_filter_matrix))
    average_diff2_X[seg] = (sum(sum(X_C_power_filter_matrix)))/(sum(sum(X_O_power_filter_matrix)))
    average_diff2_Y[seg] = (sum(sum(Y_C_power_filter_matrix)))/(sum(sum(Y_O_power_filter_matrix)))

# =============================================================================
# Plots
# =============================================================================
" Source signal X Plots "
plt.figure(1)
plt.subplot(511)
plt.plot(X_C_time, X_C_signal, label='Source 10')
plt.xlabel('Time')
plt.title('Source Signal from S1_CClean')
plt.legend()

plt.subplot(512)
plt.stem(X_C_power, label = 'Source 10' )
plt.xlabel('Frequency [Hz]')
plt.ylabel('Power')
plt.title('FFT of Source Signal from S1_CClean')
plt.axis([-1,70,0,150])
plt.legend()

plt.subplot(513)
plt.plot(hz[:150], abs(h[:150]), label="Frequency Response of Order = 5")
plt.axvline(x=8)
plt.axvline(x=13)
plt.title('Butterworth Bandpass')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Gain')
plt.legend()

plt.subplot(514)
plt.plot(X_C_time, X_C_filter, label='Source = 10')
plt.title('Filtered Source Signal from S1_CClean')
plt.xlabel('Time')
plt.legend()

plt.subplot(515)
plt.stem(X_C_power_filter,label='Source = 10' )
plt.xlabel('Frequency [Hz]')
plt.ylabel('Power')
plt.title('FFT Filtered Source Signal from S1_CClean')
plt.axis([-1,70,0,150])
plt.legend()
plt.show()
plt.savefig('figures/DFT_plot_X_timeseg15_source10.png')


" Measurement Matrix Y and Source Matrix X Plots "
plt.figure(4)

plt.subplot(221)
plt.stem(sum(Y_O_power_filter_matrixSEG), label=' Open eyes')
plt.title('FFT of Filtered $\mathbf{Measurements}$, Segment 35')
plt.ylabel('Power')
plt.xticks([])
plt.axis([-1,20,0,55000])
plt.legend()

plt.subplot(223)
plt.stem(sum(Y_C_power_filter_matrixSEG), label='Closed eyes')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Power')
plt.axis([-1,20,0,55000])
plt.legend()

plt.subplot(222)
plt.stem(sum(X_O_power_filter_matrixSEG), label='Open eyes')
plt.title('FFT of Filtered $\mathbf{Sources}$, Segment 35 ')
plt.yticks([])
plt.xticks([])
plt.axis([-1,20,0,55000])
plt.legend()

plt.subplot(224)
plt.stem(sum(X_C_power_filter_matrixSEG), label='Closed eyes')
plt.xlabel('Frequency [Hz]')
plt.yticks([])
plt.axis([-1,20,0,55000])
plt.legend()
plt.show()
plt.savefig('figures/DFT_plot_X_and_Y_matrix_timeseg35_power.png')

plt.figure(5)
plt.plot(average_diff_Y, 'ro-', label='C/O')
plt.hlines(1,0,100,label='1/1')
plt.title('C/O Relation of $\mathbf{Measurements}$ over all Segments')
plt.ylabel('C/O')
plt.xlabel('Time Segment')
plt.axis([0,100,0,2])
plt.legend()
plt.show()
plt.savefig('figures/DFT_Y_difference.png')

plt.figure(6)
plt.plot(average_diff_X, 'ro-', label='C/O')
plt.hlines(1,0,100,label='1/1')
plt.title('C/O Relation of $\mathbf{Sources}$ over all Segments')
plt.xlabel('Time Segment')
plt.ylabel('C/O')
plt.axis([0,100,0,20])
plt.legend()
plt.show()
plt.savefig('figures/DFT_X_difference.png')
