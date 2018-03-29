#!/usr/bin/env python

import sys

import rospy

from s2t.speech_recognition import SpeechRecognizer


if __name__ == '__main__':
    rospy.init_node('speech2text_engine', anonymous=True)
    try:
        recog = SpeechRecognizer()
    except SpeechRecognizer.InvalidDevice as e:
        rospy.logerr(e.message)
        sys.exit(1)
