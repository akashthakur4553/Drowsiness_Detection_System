import logging
import os
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from scipy.spatial import distance
from imutils import face_utils
from pygame import mixer
import imutils
import dlib
import twilio

logger = logging.getLogger(__name__)
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from twilio.base.exceptions import TwilioRestException
from twilio.rest import Client

audio_file = open("alarm.mp3", "rb")
audio_bytes = audio_file.read()

# Initialize the mixer for alert sound
# mixer.init()
# mixer.music.load("alarm.mp3")


# Function to calculate the eye aspect ratio
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear


# Thresholds and parameters
thresh = 0.20
frame_check = 20
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
flag = 0


def get_ice_servers():
    try:
        account_sid = "AC33632aa93a8aef5cdd18973197a2be57"
        auth_token = "204b198c1a053a12fb4fee7f2158d4e6"
    except KeyError:
        logger.warning(
            "Twilio credentials are not set. Fallback to a free STUN server from Google."
        )
        return [{"urls": ["stun:stun.l.google.com:19302"]}]

    client = Client(account_sid, auth_token)

    try:
        token = client.tokens.create()
    except TwilioRestException as e:
        st.warning(
            f"Error occurred while accessing Twilio API. Fallback to a free STUN server from Google. ({e})"
        )
        return [{"urls": ["stun:stun.l.google.com:19302"]}]

    return token.ice_servers


class DrowsinessTransformer(VideoTransformerBase):
    def __init__(self):
        self.flag = 0

    def transform(self, frame):
        global flag
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        subjects = detect(gray, 0)

        for subject in subjects:
            shape = predict(gray, subject)
            shape = face_utils.shape_to_np(shape)
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(img, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(img, [rightEyeHull], -1, (0, 255, 0), 1)

            if ear < thresh:
                self.flag += 1
                if self.flag >= frame_check:
                    cv2.putText(
                        img,
                        "****************ALERT!****************",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2,
                    )
                    cv2.putText(
                        img,
                        "****************ALERT!****************",
                        (10, 325),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2,
                    )
                    # mixer.music.play()
                    st.audio(audio_bytes, format="audio/mp3", autoplay=True, start_time=0, end_time=5)
            else:
                self.flag = 0

        return img


st.title("Drowsiness Detection System")
st.write("This app detects drowsiness in real-time using your webcam.")

ice_servers = get_ice_servers()

webrtc_streamer(
    key="drowsiness",
    video_processor_factory=DrowsinessTransformer,
    media_stream_constraints={"video": True, "audio": False},
    rtc_configuration={"iceServers": ice_servers},
)

st.write("Debug Information")
st.write(f"Streamlit version: {st.__version__}")
