import numpy as np

from collections import deque

import rospy


NORMAL_MAXIMUM = 16384
BUFFER_NP_TYPE = '<i2'  # little endian, signed short


def normalize(snd_data):
    """Average the volume out
    """
    r = snd_data * (NORMAL_MAXIMUM * 1. / max(1, np.abs(snd_data).max()))
    return r.astype(snd_data.dtype)


def add_silence(snd_data, rate, seconds):
    """Adds silence of given length to the start and end of a chunk.

    This prevents some players from skipping the first few frames.
    :param snd_data: numpy array
        sound chunk
    :param rate: int
        sampling rate
    :param seconds: float
        length of the silence to add
    """
    zeros = np.zeros((int(seconds * rate),), dtype=snd_data.dtype)
    return np.hstack([zeros, snd_data, zeros])


class SilenceDetector(object):
    """Implements silence detection on chunks of sound."""

    def is_silent(self, snd_data):
        """
        Returns 'True' if all the data is below the 'silent' threshold.
        """
        return np.abs(snd_data).max() < self.threshold

    def trim(self, snd_data):
        """Trim the blank spots at the start and end."""
        raise NotImplementedError

    def reset_average(self):
        pass

    def update_average(self, chunk):
        pass


class StaticSilenceDetector(SilenceDetector):

    is_static = True

    def __init__(self, rate, threshold):
        self.rate = rate
        self.threshold = threshold

    def trim(self, snd_data):
        non_silent = (np.abs(snd_data) <= self.threshold).nonzero()[0]
        if len(non_silent) == 0:
            return snd_data[0:0]  # Empty array
        else:
            return snd_data[non_silent[0]:non_silent[-1]]


class DynamicSilenceDetector(SilenceDetector):

    is_static = False

    def __init__(self, rate, dynamic_threshold_percentage=50,
                 min_average_volume=1., n_average=10):
        self.rate = rate
        self.dyn_thr_ratio = dynamic_threshold_percentage / 100.
        self.min_avg = min_average_volume
        self._vol_q = deque([], maxlen=n_average)
        self.reset_average()

    @property
    def average_volume(self):
        if len(self._vol_q) == 0:
            return self.min_avg
        else:
            return max(self.min_avg, sum(self._vol_q) * 1. / len(self._vol_q))

    def reset_average(self):
        self._vol_q.clear()
        # Note: storing the sum and incrementing/decrementing on update
        # would be more efficient but this is simpler for now.

    def update_average(self, chunk):
        self._vol_q.append(max(np.abs(chunk)))

    @property
    def threshold(self):
        """(100+x)% louder than the avg volume."""
        return self.average_volume * (1 + self.dyn_thr_ratio)


class SpeechDetector:
    """
    Dynamic thresholding:
        Before audio is being considered part of a sentence, peak_count
        is used to count how many consecutive frames have been above the
        dynamic threshold volume. Once peak_count is over the specified
        frame number from ros param, we consider the sentence started and
        lock the value of avg volume to maintain the standard thresholding
        standard throughout the sentence. Whenever receiving a frame that
        has volume that is too low, we increase num_silent. When num_silent
        exceeds ten, we consider the sentence finished.

    :param threshold: float
        Static or dynamic threshold. Interpreted as a percentage when dynamic.
    :param n_silent: int
        Number of silent chunks to end detected utterance.
    """

    def __init__(self, rate, threshold, dynamic_threshold=False,
                 dynamic_threshold_frame=3, chunk_size=None,
                 min_average_volume=0., n_silent=10):
        self.rate = rate
        if dynamic_threshold:
            self.silence_detect = DynamicSilenceDetector(
                self.rate, threshold, min_average_volume=min_average_volume)
        else:
            self.silence_detect = StaticSilenceDetector(self.rate, threshold)
        if chunk_size is None:
            chunk_size = self.rate // 10
        self.chunk_size = chunk_size
        self.dyn_thr_frame = dynamic_threshold_frame
        self.max_n_silent = n_silent
        self.reset()

    def reset(self):
        self.silence_detect.reset_average()
        self.n_silent = 0
        self.n_peaks = 0
        self.chunks = []
        self.in_utterance = False
        self.start_time = None

    def treat_chunk(self, chunk):
        silent = self.silence_detect.is_silent(chunk)
        # Print average for dynamic threshold
        # TODO: should be a debug
        if not self.silence_detect.is_static:
            rospy.logdebug("[AVG_VOLUME,VOLUME] = {}, {}".format(
                self.silence_detect.average_volume, max(np.abs(chunk))))
        if not silent and not self.in_utterance:
            # Check whether to start collecting utterance
            if not self.silence_detect.is_static:  # TODO: Why only for dynamic?
                self.n_peaks += 1
                self.silence_detect.update_average(chunk)
                self.chunks.append(chunk)
            if (self.silence_detect.is_static or
                    self.n_peaks >= self.dyn_thr_frame):
                rospy.logdebug('collecting audio segment')
                self.start_time = rospy.get_rostime()
                self.in_utterance = True
        if silent and not self.in_utterance:
            self.silence_detect.update_average(chunk)
            self.chunks = []
            self.n_peaks = 0
        if self.in_utterance:
            self.chunks.append(chunk)
            if silent:
                self.n_silent += 1
            else:
                self.n_silent = 0

    @property
    def found(self):
        return self.n_silent > self.max_n_silent

    def get_next_utter(self, stream, start_callback, end_callback):
        """
        Main function for capturing audio.
        Parameters:
            stream: our pyaudio client
            min_avg_volume: helps thresholding in quiet environments
            pub_screen: publishes status messages to baxter screen
        """
        stream.start_stream()  # TODO: Why not record during recognition
        self.reset()
        previously = False

        while not self.found:
            # main loop for audio capturing
            if self.in_utterance and not previously:
                start_callback()
            previously = self.in_utterance

            if rospy.is_shutdown():
                return None, None, None

            snd_data = np.frombuffer(
                stream.read(self.chunk_size),
                dtype=BUFFER_NP_TYPE)
            self.treat_chunk(snd_data)

        stream.stop_stream()
        end_time = rospy.get_rostime()

        end_callback()

        r = normalize(np.hstack(self.chunks))
        if self.silence_detect.is_static:
            r = self.silence_detect.trim(r)
        r = add_silence(r, self.rate, 1)
        assert(isinstance(self.start_time, rospy.rostime.Time))
        assert(isinstance(end_time, rospy.rostime.Time))
        return r, self.start_time, end_time
