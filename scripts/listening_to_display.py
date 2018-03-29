from threading import Lock

import rospy
from std_msgs.msg import String

from s2t.speech_recognition import SpeechRecognizer
from s2t.speech_recognition.msg import event
from s2t.speech_recognition.msg import status


class ListeningToDisplay(object):
    # added status.msg (in ros_speech2txt): bool listen, bool process


    PERIOD = .1
    MESSAGE = status(); 
    # To set message duration on the display node
    DURATION = 1
    DURATION_PARAM = 'baxter_display/speech_duration'

    def __init__(self, display_topic):
        rospy.init_node('Display listening/processing')
        rospy.set_param(self.DURATION_PARAM, self.DURATION)
        self.sub = rospy.Subscriber(SpeechRecognizer.TOPIC_BASE + '/log',
                                    event, self._event_cb)
        self.pub = rospy.Publisher('status_topic', status, queue_size = 2)
        self._listening = False
        self._processing = False
        self._lock = Lock()

    def run(self):
        self.running = True
        self.last_msg = rospy.Time.now()
        while self.running and not rospy.is_shutdown():
            if (self.listening or self.processing) and (
                    rospy.Time.now() - self.last_msg) > self.DURATION:
                MESSAGE.listen = self.listening
                MESSAGE.process = self.processing
                self.pub.publish(self.MESSAGE)

    @property
    def listening(self):
        with self._lock:
            return self._listening

    def processing(self):
        with self._lock:
            return self._processing

    @listening.setter
    def listening(self, b):
        with self._lock:
            self._listening = b

    @processing.setter
    def processing(self, b):
        with self._lock:
            self._listening = b

    def _event_cb(self, msg):
        if msg.event == event.STARTED:
            self.listening = True
            self.processing = False
        elif msg.event == event.STOPPED:
            self.listening = False
            self.processing = True


if __name__ == '__main__':
ListeningToDisplay('/svox_tts/speech_output').run()