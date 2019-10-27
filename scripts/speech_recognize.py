#!/usr/bin/env python

import rospy
import numpy as np
from google.cloud import speech_v1p1beta1 as speech
from google.cloud.speech_v1p1beta1 import enums
from google.cloud.speech_v1p1beta1 import types

from std_msgs.msg import String, Header, Time
from ros_speech2text.msg import Utterance, Transcript

NORMAL_MAXIMUM = 16384

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

def callback(msg, cb_args):
    (speech_client, pub_transcript, output_stream, keywords) = cb_args

    sample_rate = msg.audio_config.sample_rate
    chunk = msg.audio_chunk.chunk

    if msg.audio_config.sample_width != 2:
        raise Exception('Width ' + str(sample_width) + ' cannot be handled.')

    chunk = np.fromstring(chunk, dtype=np.int16)
    # normalize
    chunk = normalize(chunk)
    # add silence
    chunk = add_silence(chunk, sample_rate, 1)

    chunk = chunk.tostring()

    config = types.RecognitionConfig(
        encoding='LINEAR16',
        sample_rate_hertz=sample_rate,
        language_code='en-US',
        enable_automatic_punctuation=True,
        speech_contexts=[types.SpeechContext(phrases=keywords,)]
    )

    audio = types.RecognitionAudio(content=chunk)
    response = speech_client.recognize(config, audio)
    if response.results:
        transcript = response.results[0].alternatives[0].transcript
        confidence = response.results[0].alternatives[0].confidence
        # pub_transcript.publish(transcript, msg.start_time, msg.duration, msg.index, confidence)
        pub_transcript.publish(transcript, confidence, msg.start_time, msg.duration, msg.index)
        rospy.loginfo('From: ' + output_stream)
        rospy.loginfo('Transcript: ' + transcript)
    else:
        rospy.loginfo('Speech not recognized.')

if __name__ == '__main__':
    rospy.init_node('speech_recognize', anonymous = True)
    node_name = rospy.get_name()

    input_stream = rospy.get_param(node_name + '/input_stream')

    output_stream = rospy.get_param(node_name + '/output_stream')

    keywords = rospy.get_param(node_name + '/keywords')
    print keywords
    print type(keywords)

    pub_transcript = rospy.Publisher(output_stream + '/transcript', Transcript)
    rospy.loginfo('Publishing transcripts to {}'.format(output_stream + '/transcript'))

    client = speech.SpeechClient()

    rospy.Subscriber(input_stream + '/complete', Utterance, callback, (client, pub_transcript, output_stream, keywords))
    rospy.spin()