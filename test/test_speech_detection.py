#!/usr/bin/env python
PKG = 'ros_speech2text'

import unittest
from unittest import TestCase
from std_msgs.msg import String
import numpy as np
import rospkg
import rospy
import os

from ros_speech2text.msg import transcript
from s2t.speech_detection import (StaticSilenceDetector,
                                  DynamicSilenceDetector,
                                  normalize, NORMAL_MAXIMUM,
                                  add_silence)


class TestNormalize(TestCase):

    @staticmethod
    def test_normalize_ones():
        a = np.ones((5,))
        b = NORMAL_MAXIMUM * a
        np.testing.assert_array_equal(normalize(a), b)

    @staticmethod
    def test_normalize_twos():
        a = 2 * np.ones((5,))
        b = np.ones((5,)) * NORMAL_MAXIMUM
        np.testing.assert_array_equal(normalize(a), b)

    @staticmethod
    def test_normalize():
        a = np.array([10, 1, 0])
        b = np.array([NORMAL_MAXIMUM, NORMAL_MAXIMUM // 10, 0])
        np.testing.assert_array_equal(normalize(a), b)

    @staticmethod
    def test_normalize_already_normal():
        a = np.array([NORMAL_MAXIMUM, NORMAL_MAXIMUM // 10, 0])
        np.testing.assert_array_equal(normalize(a), a)

    def test_normalize_result_is_int(self):
        a = (100 * np.random.random((5,))).astype(np.int16)
        self.assertEqual(normalize(a).dtype, np.int16)


class TestAddSilence(TestCase):
    def test_adds_silence_front(self):
        pass
    def test_adds_silence_end(self):
        pass

class TestUtteranceDetection(TestCase):
    def test_silent_array_with_clean_average_no_utterance(self):
        # there should be no utterance detected
        pass
    def test_silent_array_with_random_average_no_utterance(self):
        # there should be no utterance detected
        pass
    def test_constant_array_with_clean_average_is_utterance(self):
        # whole array should be classified as utterance
        pass
    def test_constant_array_with_high_average_no_utterance(self):
        # there should be no utterance detected
        pass
    def test_sin_array_with_clean_average_is_utterance(self):
        # a part of the array should be classified utterance
        pass

class TestStaticSilenceDetector(TestCase):

    def test_silent_is_silent(self):
        sd = StaticSilenceDetector(100, 1.)
        a = np.array([0, 0, 0, 0, 0], dtype=np.int16)
        self.assertTrue(sd.is_silent(a))

    def test_above_static(self):
        sd = StaticSilenceDetector(100, 1.)
        a = np.array([0, 0, 2, 0, 0], dtype=np.int16)
        self.assertFalse(sd.is_silent(a))


class TestDynamicSilenceDetector(TestCase):

    def test_silent_is_silent_dynamic(self):
        sd = DynamicSilenceDetector(100, min_average_volume=1.)
        a = np.array([0, 0, 0, 0, 0], dtype=np.int16)
        self.assertTrue(sd.is_silent(a))

    def test_average_volume_is_min_on_empty(self):
        sd = DynamicSilenceDetector(100, 50., min_average_volume=.1)
        self.assertEqual(sd.average_volume, .1)

    def test_average_volume_is_not_min(self):
        sd = DynamicSilenceDetector(100, 50., min_average_volume=.1)
        sd.update_average(np.array([0, 1], dtype=np.int16))
        sd.update_average(np.array([0, 0], dtype=np.int16))
        self.assertEqual(sd.average_volume, .5)

    def test_above_static_but_not_dynamic(self):
        sd = DynamicSilenceDetector(100, 50.)
        sd.update_average(np.array([1], dtype=np.int16))
        sd.update_average(np.array([2], dtype=np.int16))
        # Hence sd.avg_volume == 1.5
        a = np.array([0, 0, 2, 0, 0], dtype=np.int16)
        self.assertTrue(sd.is_silent(a))

    def test_above_dynamic_but_not_static(self):
        sd = DynamicSilenceDetector(100, 50.)
        sd.update_average(np.array([1], dtype=np.int16))
        sd.update_average(np.array([1], dtype=np.int16))
        # Hence sd.avg_volume == 1
        a = np.array([0, 0, 2, 0, 0], dtype=np.int16)
        self.assertFalse(sd.is_silent(a))

    def test_reset_average(self):
        sd = DynamicSilenceDetector(100, 50., min_average_volume=1.5)
        sd.update_average(np.array([2], dtype=np.int16))
        sd.update_average(np.array([2], dtype=np.int16))
        # Hence sd.avg_volume == 2
        sd.reset_average()
        self.assertEqual(sd.average_volume, 1.5)

class TestSpeechAPI(TestCase):
    def setUp(self):
        
        p = rospkg.RosPack()
        package_path = p.get_path('ros_speech2text')
        self.audio_path = os.path.join(package_path,'test','test_audio')
        self.result = None

    # def tearDown(self):
    #     self.pub.unregister()

    def callback1(self,recog_result):
        print "results received"
        self.result = recog_result.transcript

    def test_sentence1(self):
        self.pub= rospy.Publisher('/test_input',String,queue_size=10)
        file_path = os.path.join(self.audio_path,'sentence0.wav')
        print file_path
        self.pub.publish(file_path)
        rospy.Subscriber('ros_speech2text/user_output',transcript,self.callback1)
        while self.result==None:
            print "waiting for result"
            rospy.sleep(1)
        print self.result
        self.assertEqual(self.result,'good morning Baxter how are you doing today')


if __name__ == '__main__':
    import rostest
    rospy.init_node('test_ros_speech2text')
    rostest.rosrun(PKG, 'test_speech_detection', TestSpeechAPI)
