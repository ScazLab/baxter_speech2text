import os
import io
import csv
import shutil
from struct import pack
from threading import Thread

import wave
import pyaudio

import rospy
from std_msgs.msg import String, Header
from ros_speech2text.msg import transcript, event

from .speech_detection import SpeechDetector

from deepspeech.model import Model
import scipy.io.wavfile as wav
from timeit import default_timer as timer

FORMAT = pyaudio.paInt16

# TODO: move this to some configuration file maybe.
# Beam width used in the CTC decoder when building candidate transcriptions
BEAM_WIDTH = 500

# The alpha hyperparameter of the CTC decoder. Language Model weight
LM_WEIGHT = 1.75

# The beta hyperparameter of the CTC decoder. Word insertion weight (penalty)
WORD_COUNT_WEIGHT = 1.00

# Valid word insertion weight. This is used to lessen the word insertion penalty
# when the inserted word is part of the vocabulary
VALID_WORD_COUNT_WEIGHT = 1.00


# These constants are tied to the shape of the graph used (changing them changes
# the geometry of the first layer), so make sure you use the same constants that
# were used during training

# Number of MFCC features to use
N_FEATURES = 26

# Size of the context window used for producing timesteps in the input vector
N_CONTEXT = 9


def list_audio_devices(pyaudio_handler):
    device_list = [pyaudio_handler.get_device_info_by_index(i)['name']
                   for i in range(pyaudio_handler.get_device_count())]
    rospy.logdebug('Available devices:' + ''.join(
        ['\n  - [%d]: %s' % d for d in enumerate(device_list)]))
    rospy.set_param('/ros_speech2text/available_audio_device', device_list)
    return device_list


class SpeechRecognizer(object):

    TOPIC_BASE = '/speech_to_text'

    class InvalidDeviceID(ValueError):
        pass

    def __init__(self):
        self._init_history_directory()
        self.node_name = rospy.get_name()
        self.print_level = rospy.get_param('/print_level', 0)
        self.pub_transcript = rospy.Publisher(
            self.TOPIC_BASE + '/transcript', transcript, queue_size=10)
        self.pub_text = rospy.Publisher(
            self.TOPIC_BASE + '/text', String, queue_size=10)
        self.pub_event = rospy.Publisher(
            self.TOPIC_BASE + '/log', event, queue_size=10)
        self.sample_rate = rospy.get_param(self.node_name + '/audio_rate', 16000)
        self.async = rospy.get_param(self.node_name + '/async_mode', True)
        dynamic_thresholding = rospy.get_param(
            self.node_name + '/enable_dynamic_threshold', True)
        if not dynamic_thresholding:
            threshold = rospy.get_param(self.node_name + '/audio_threshold', 700)
        else:
            threshold = rospy.get_param(
                self.node_name + '/audio_dynamic_percentage', 50)
        self.speech_detector = SpeechDetector(
            self.sample_rate,
            threshold,
            dynamic_threshold=dynamic_thresholding,
            dynamic_threshold_frame=rospy.get_param(
                self.node_name + '/audio_dynamic_frame', 3),
            min_average_volume=rospy.get_param(
                self.node_name + '/audio_min_avg', 100),
            n_silent=rospy.get_param(
                self.node_name + '/n_silent_chunks', 10),
        )
        self._init_stream()
        self._init_csv()
        
        model = rospy.get_param(self.node_name + '/deepspeech_model', "~/deepspeech/models/output_graph.pb")
        alphabet = rospy.get_param(self.node_name + '/deepspeech_alphabet', "~/deepspeech/models/alphabet.txt")
        lm = rospy.get_param(self.node_name + '/deepspeech_language_model', "~/deepspeech/models/lm.binary")
        trie = rospy.get_param(self.node_name + '/deepspeech_trie', "~/deepspeech/models/trie")

        print('Loading model from file %s' % (model))
        model_load_start = timer()
        self.ds = Model(model, N_FEATURES, N_CONTEXT, alphabet, BEAM_WIDTH)
        model_load_end = timer() - model_load_start
        print('Loaded model in %0.3fs.' % (model_load_end))

        if lm is not "" and trie is not "":
            print('Loading language model from files %s %s' % (lm, trie))
            lm_load_start = timer()
            self.ds.enableDecoderWithLM(alphabet, lm, trie, LM_WEIGHT,
                                   WORD_COUNT_WEIGHT, VALID_WORD_COUNT_WEIGHT)
            lm_load_end = timer() - lm_load_start
            print('Loaded language model in %0.3fs.' % (lm_load_end))

        self.run()

    def _init_history_directory(self):
        param = rospy.get_param('/ros_speech2text/speech_history',
                                '~/.ros/ros_speech2text/speech_history')
        self.history_dir = os.path.expanduser(os.path.join(param, str(os.getpid())))
        if not os.path.isdir(self.history_dir):
            os.makedirs(self.history_dir)

    def _init_stream(self):
        self.pa_handler = pyaudio.PyAudio()
        device_list = list_audio_devices(self.pa_handler)
        input_idx = rospy.get_param(self.node_name + '/audio_device_idx', None)
        input_name = rospy.get_param(self.node_name + '/audio_device_name', None)
        if input_idx is None:
            input_idx = self.pa_handler.get_default_input_device_info()['index']
            if input_name is not None:
                try:
                    # use first found for name
                    input_idx = [input_name.lower() in d.lower()
                                 for d in device_list
                                 ].index(True)
                except ValueError:
                    rospy.logerr(
                        "No device found for name '%s', falling back to default."
                        % input_name)
        try:
            rospy.loginfo("{} using device: {}".format(
                self.node_name,
                self.pa_handler.get_device_info_by_index(input_idx)['name'])
            )
            self.stream = self.pa_handler.open(
                format=FORMAT, channels=1, rate=self.sample_rate, input=True,
                start=False, input_device_index=input_idx, output=False,
                frames_per_buffer=self.speech_detector.chunk_size)
        except IOError:
            self.terminate()
            raise self.InvalidDeviceID(
                'Invalid device ID: {}. Available devices listed in rosparam '
                '/ros_speech2text/available_audio_device'.format(input_idx))
        self.sample_width = self.pa_handler.get_sample_size(FORMAT)

    def _init_csv(self):
        self.csv_file = open(os.path.join(self.history_dir, 'transcript'), 'wb')
        self.csv_writer = csv.writer(self.csv_file, delimiter=' ',)
        self.csv_writer.writerow(['start', 'end', 'duration', 'transcript', 'confidence'])

    def run(self):
        sn = 0
        while not rospy.is_shutdown():
            aud_data, start_time, end_time = self.speech_detector.get_next_utter(
                self.stream, *self.get_utterance_start_end_callbacks(sn))
            if aud_data is None:
                rospy.loginfo("No more data, exiting...")
                break
            self.record_to_file(aud_data, sn)
            transc = self.recog(sn)
            confidence = 1.0 # TODO: can we get this???
            self.utterance_decoded(sn, transc, confidence, start_time, end_time)
            sn += 1
        self.terminate()

    def terminate(self, exitcode=0):
        self.stream.close()
        self.pa_handler.terminate()
        self.csv_file.close()
        if rospy.get_param(rospy.get_name() + '/cleanup', True):
            shutil.rmtree(self.history_dir)

    def utterance_start(self, utterance_id):
        if self.print_level > 1:
            rospy.loginfo('Utterance started')
        self.pub_event.publish(
            self.get_event_base_message(event.STARTED, utterance_id))

    def utterance_end(self, utterance_id):
        if self.print_level > 1:
            rospy.loginfo('Utterance completed')
        self.pub_event.publish(
            self.get_event_base_message(event.STOPPED, utterance_id))

    def get_utterance_start_end_callbacks(self, utterance_id):
        def start():
            self.utterance_start(utterance_id)

        def end():
            self.utterance_end(utterance_id)

        return start, end

    def utterance_decoded(self, utterance_id, transcription, confidence,
                          start_time, end_time):
        transcript_msg = self.get_transcript_message(transcription, confidence,
                                                     start_time, end_time)
        event_msg = self.get_event_base_message(event.DECODED, utterance_id)
        event_msg.transcript = transcript_msg
        if self.print_level > 0:
            rospy.loginfo("{} [confidence: {}]".format(transcription, confidence))
        self.pub_transcript.publish(transcript_msg)
        self.pub_text.publish(transcription)
        self.pub_event.publish(event_msg)
        self.csv_writer.writerow([
            start_time, end_time, transcript_msg.speech_duration,
            transcription, confidence])

    def utterance_failed(self, utterance_id, start_time, end_time):
        if self.print_level > 1:
            rospy.loginfo("No good results returned!")
        transcript_msg = self.get_transcript_message("", 0., start_time, end_time)
        event_msg = self.get_event_base_message(event.FAILED, utterance_id)
        event_msg.transcript = transcript_msg
        self.pub_event.publish(event_msg)

    def get_transcript_message(self, transcription, confidence, start_time,
                               end_time):
        msg = transcript()
        msg.start_time = start_time
        msg.end_time = end_time
        msg.speech_duration = end_time - start_time
        msg.received_time = rospy.get_rostime()
        msg.transcript = transcription
        msg.confidence = confidence
        return msg

    def get_event_base_message(self, evt, utterance_id):
        msg = event()
        msg.header = Header()
        msg.header.stamp = rospy.Time.now()
        msg.event = evt
        msg.utterance_id = utterance_id
        msg.audio_path = self.utterance_file(utterance_id)
        return msg

    def utterance_file(self, utterance_id):
        file_name = 'utterance_{}.wav'.format(utterance_id)
        return os.path.join(self.history_dir, file_name)

    def record_to_file(self, data, utterance_id):
        """Saves audio data to a file"""
        data = pack('<' + ('h' * len(data)), *data)
        path = self.utterance_file(utterance_id)
        wf = wave.open(path, 'wb')
        wf.setnchannels(1)
        wf.setsampwidth(self.sample_width)
        wf.setframerate(self.sample_rate)
        wf.writeframes(data)
        wf.close()
        rospy.logdebug('File saved to {}'.format(path))

    def recog(self, utterance_id):
        """
        Constructs a recog operation with the audio file specified by sn
        The operation is an asynchronous api call
        """
        context = rospy.get_param(self.node_name + '/speech_context', [])
        path = self.utterance_file(utterance_id)

        fs, audio = wav.read(path)
        audio_length = len(audio) * ( 1 / 16000)
        assert fs == 16000, "Only 16000Hz input WAV files are supported for now!"

        print('Running inference.')
        inference_start = timer()
        stt_result = self.ds.stt(audio, fs)
        inference_end = timer() - inference_start
        print('Inference took %0.3fs for %0.3fs audio file.' % (inference_end, audio_length))

        return stt_result
