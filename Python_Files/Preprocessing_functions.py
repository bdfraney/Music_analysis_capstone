import numpy as np
import matplotlib.pyplot as plt
import scipy

import librosa
from librosa.feature import chroma_stft


def plot_mfc(song, title):
	y, sr = librosa.load(song, sr=42000, res_type='kaiser_fast')

	# Let's make and display a mel-scaled power (energy-squared) spectrogram
	s = librosa.feature.melspectrogram(y, sr=sr, n_mels=128)

	# Convert to log scale (dB). We'll use the peak power as reference.
	log_s = librosa.amplitude_to_db(s, ref=np.max)

	# Make a new figure
	plt.figure(figsize=(18, 4))

	# Display the spectrogram on a mel scale
	# sample rate and hop length parameters are used to render the time axis
	librosa.display.specshow(log_s, sr=sr, x_axis='time', y_axis='mel')

	# Put a descriptive title on the plot
	plt.title(title + ' mel power spectrogram')

	# draw a color bar
	plt.colorbar(format='%+02.0f dB')

	# Make the figure layout compact
	plt.tight_layout()
	pass


def plot_chroma(file, title):
	# Load in the song using kaiser_fast to speed up loading
	x, sr = librosa.load(file, sr=42000, res_type='kaiser_fast')

	s = np.abs(librosa.stft(x, n_fft=4096)) ** 2

	# Generating chromagram
	chroma = chroma_stft(S=s, sr=sr)

	plt.figure(figsize=(18, 4))

	librosa.display.specshow(chroma, y_axis='chroma', x_axis='time')

	plt.colorbar()
	plt.title(title + ' Chromagram')
	plt.tight_layout()
	plt.show()
	pass


# noinspection PyBroadException
def mfccs_parser(file: str, sample_rate: int = 42000) -> np.array:
	"""
	:type sample_rate: int
	:param sample_rate: The sample rate used when loading audio file.  Default is 42000 which is the typical sample
	rate for commercial music files.
	:param file: Path to an audio file
	:return: NumPy array of the Mel-frequency cepstral coefficients
	:rtype: NumPy array
	"""
	try:

		# here kaiser_fast is a technique used for faster extraction (though it does negatively affect quality)
		x, sample_rate = librosa.load(file, sr=sample_rate, res_type='kaiser_fast')

		# we extract mfcc feature from data, Use the mean so that scale isn't an issue.
		mfccs = np.mean(librosa.feature.mfcc(y=x, sr=sample_rate, n_mfcc=40).T, axis=0)

	except Exception:
		print("Error encountered while parsing file: ", file)
		return None, None

	#     feature = mfccs
	#     label = row

	return mfccs


def enhanced_chroma_parser(file):
	try:

		# Load in the song using kaiser_fast to speed up loading
		y, sr = librosa.load(file, sr=42000, res_type='kaiser_fast')

		y_harm = librosa.effects.harmonic(y=y, margin=8)
		chroma_os_harm = librosa.feature.chroma_cqt(y=y_harm, sr=sr, bins_per_octave=12 * 3)

		chroma_filter = np.minimum(chroma_os_harm,
		                           librosa.decompose.nn_filter(chroma_os_harm, aggregate=np.median, metric='cosine'))

		chroma_smooth = np.mean(scipy.ndimage.median_filter(chroma_filter, size=(1, 9)).T, axis=0)

	except:
		print("Error encountered while parsing file: ", file)
		return None, None

	#     feature = mfccs
	#     label = row

	return chroma_smooth


def get_mfc(file, sample_rate=42000):
	try:

		y, sr = librosa.load(file, sr=sample_rate, res_type='kaiser_fast')

		# Let's make and display a mel-scaled power (energy-squared) spectrogram
		s = librosa.feature.melspectrogram(y, sr=sr, n_mels=128)

		# Convert to log scale (dB). We'll use the peak power as reference.
		mfc = librosa.amplitude_to_db(s, ref=np.max)

		return mfc

	except:
		print("Error encountered while parsing file: ", file)
		pass


def mfcc_chroma_parser(file, sample_rate=22050, n_mels=114):
	"""
	Combination of the mfcc parser and the enhance chroma.

	:param file: String for file path
	:param sample_rate: sample rate for loading the audio time series.
	:param n_mels: number of mel coefficients.
	"""

	try:
		# Load in the song using kaiser_fast to speed up loading
		y = librosa.load(file, sr=sample_rate, res_type='kaiser_fast')[0]

		y_harm = librosa.effects.harmonic(y=y, margin=8)

		chroma_os_harm = librosa.feature.chroma_cqt(y=y_harm, sr=sample_rate, bins_per_octave=12 * 3)

		chroma_filter = np.minimum(chroma_os_harm, librosa.decompose.nn_filter(chroma_os_harm,
		                                                                       aggregate=np.median,
		                                                                       metric='cosine'))

		chroma_smooth = np.mean(scipy.ndimage.median_filter(chroma_filter, size=(1, 9)).T, axis=0)

		mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sample_rate, n_mfcc=n_mels).T, axis=0)

		feature = np.hstack((mfccs, chroma_smooth))

		return feature

	except:
		print("Error encountered while parsing file: ", file)
		pass
