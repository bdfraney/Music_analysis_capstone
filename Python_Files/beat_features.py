import librosa
import numpy as np


def beat_feat_parser(file: str, n_mfccs: int = 40, sr: int = 22050, hop_length: int = 512, start_bpm: float = 150.0):
	"""

	:param file: String; name of the file to extract features from.

	:param n_mfccs: Int; number of mel-frequency cepstral coefficients to generate. Default is 40 because that's what I used.

	:param sr: Int; the sample rate, set to 22050 for default (42000 recommended music but takes longer and the
	resulting array is significantly bigger.

	:param hop_length: Int; The number of samples between successive frames, default 512. At 22050 Hz, 512 samples ~= 23ms

	:param start_bpm: Float; Estimate for the tempo in bpm for the song.  A guess of 120 is usually good for most
	types of music however I chose 150 to better fit my data.

	:return: Stacked numpy array with dimensions (12 + (n_mfccs * 2), ).  The 12 comes from the beat_chroma representing
	the 12 pitch classes (C, D, E, F, G, A, B).  Each component is the corresponding arithmetic mean of the intensity at
	each frame of the audio file (proportional to length of audio).
	"""
	try:

		# Kaiser fast speeds up loading the audio time series at the cost of some quality.
		y, sr = librosa.load(file, sr=sr, res_type="kaiser_fast")

		# Separate harmonics and percussives into two waveforms
		y_harmonic, y_percussive = librosa.effects.hpss(y)

		# Beat track on the percussive signal
		tempo, beat_frames = librosa.beat.beat_track(y=y_percussive,
		                                             sr=sr, start_bpm=start_bpm)

		# Compute MFCC features from the raw signal
		mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop_length, n_mfcc=n_mfccs)

		# And the first-order differences (delta features)
		mfcc_delta = librosa.feature.delta(mfcc)

		# Stack and synchronize between beat events
		# This time, we'll use the mean value (default) instead of median
		beat_mfcc_delta = librosa.util.sync(np.vstack([mfcc, mfcc_delta]),
		                                    beat_frames)

		# Aggregating feature to condense into single subfeature vector.
		beat_mfcc_mean = np.mean(beat_mfcc_delta.T, axis=0)

		# Compute chroma features from the harmonic signal
		chromagram = librosa.feature.chroma_cqt(y=y_harmonic,
		                                        sr=sr)

		# Aggregate chroma features between beat events
		# We'll use the median value of each feature between beat frames
		beat_chroma = librosa.util.sync(chromagram,
		                                beat_frames,
		                                aggregate=np.median)

		beat_chroma_mean = np.mean(beat_chroma.T, axis=0)

		# Finally, stack all beat-synchronous features together
		beat_features = np.hstack([beat_chroma_mean, beat_mfcc_mean])

	except:
		print("Error encountered while parsing file: ", file)
		return None, None

	return beat_features


def static_tempo(file: str, bpm_estimate: float = 120.0):
	"""
    Estimate the static tempo for the song in beats per minute (bpm).  This can be the perceived tempo but often
    tempo can change throughout the song (electronic/dance music being a notable exception as those songs usually
    maintain a constant tempo) so the static tempo may not be a useful estimate.

	:param file: Str; file to be parsed.

	:param bpm_estimate: initial bpm estimate.  The accuracy of this function is highly dependent on this so some
	prior knowledge is useful.  Tracks are typically in the 100-120 bpm range so that is normally a safe guess.
	However, if it is significantly higher or lower (example: Dubstep is over 160 bpm and 120 would return a
	significantly lower estimate)

	:return: Int; static tempo estimate in bpm.
	"""
	y, sr = librosa.load(file, sr=22050, res_type="kaiser_fast")

	# Separate harmonics and percussives into two waveforms (harmoics not used for this function)
	y_harmonic, y_percussive = librosa.effects.hpss(y)

	onset_env = librosa.onset.onset_strength(y_percussive, sr=sr)

	tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr, start_bpm=bpm_estimate)

	return tempo