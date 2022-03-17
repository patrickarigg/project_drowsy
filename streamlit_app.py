# Import for streamlit
import streamlit as st
import av
import time
import mediapipe as mp
import simpleaudio as sa
from streamlit_webrtc import (
    RTCConfiguration,
    VideoProcessorBase,
    WebRtcMode,
    webrtc_streamer,
)

#Import for handling image
from cv2 import cv2
from PIL import Image
# Models and preprocessing
from project_drowsy.predict import make_prediction, get_models
from tensorflow.keras.models import load_model
import os

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

@st.cache(allow_output_mutation=True)
def get_models():
    local_path = f"{os.getcwd()}/project_drowsy/models/"
    face_model = load_model(os.path.join(local_path, 'project_drowsy_models_face_model.h5'))
    eye_model = load_model(os.path.join(local_path, 'project_drowsy_models_eye_model.h5'))
    return face_model, eye_model

@st.cache(allow_output_mutation=True)
def get_mediapipe_models():
    mp_eye_detect_model = mp.solutions.face_mesh
    face_mesh = mp_eye_detect_model.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mp_face_detect_model = mp.solutions.face_detection
    face_detection = mp_face_detect_model.FaceDetection(
        model_selection=1, min_detection_confidence=0.5)
    return face_detection, face_mesh


#Main intelligence of the file, class to launch a webcam, detect faces, then detect drowsiness and output probability for drowsiness
def app_drowsiness_detection():
    class DrowsinessPredictor(VideoProcessorBase):

        def __init__(self) -> None:

            self.face_model, self.eye_model = get_models()
            self.face_detection, self.face_mesh = get_mediapipe_models()
            self.wave_obj = sa.WaveObject.from_wave_file('airhorn.wav')
            self.drowsy_counter = 0
            self.play_sound_alert = False
            self.play = None
            self.playing = False

        def draw_and_predict(self, image):

            preprocess_params = dict(webcam=image,
                                    image_size=145,
                                    predict=True)

            try:
                # FIND FACE & EYES,GET BOUNDING BOX COORDS AND MAKE PREDICTION
                prediction, probs, face_coords, left_eye_coords, right_eye_coords = make_prediction(
                    self.face_model, self.eye_model, self.face_detection,
                    self.face_mesh, **preprocess_params)

                print(prediction)

                # draw eye bounding boxes using co-ordinates of the bounding box (from preprocessing)
                ## draw left eye box
                xmin_l,xmax_l,ymin_l,ymax_l = left_eye_coords
                cv2.rectangle(image, (xmax_l,ymax_l), (xmin_l,ymin_l),
                            color=(0, 255, 0), thickness=2)
                ## draw right eye box
                xmin_r, xmax_r, ymin_r, ymax_r = right_eye_coords
                cv2.rectangle(image, (xmax_r, ymax_r), (xmin_r, ymin_r),
                            color=(0, 255, 0),
                            thickness=2)

                ##draw face box
                xmin, xmax, ymin, ymax = face_coords
                cv2.rectangle(image, (xmax, ymax), (xmin, ymin),
                            color=(0, 255, 0),
                            thickness=3)


                # # Put text on image

                drowsy = (('r_closed' in prediction) and ('l_closed' in prediction)) or ("yawn" in prediction)

                if drowsy:
                    self.drowsy_counter += 1
                    if self.drowsy_counter >= 8:
                        text = "WARNING! DROWSY DRIVER!"
                        colour = (255, 0, 0)
                        #self.play_sound_alert = True #COMMENT THIS LINE OUT TO REMOVE SOUND

                    else:
                        text = "DRIVER IS ALERT"
                        colour = (0, 255, 0)

                    cv2.putText(image, text, (40, 40),
                            cv2.FONT_HERSHEY_PLAIN, 2, colour, 2)
                else:
                    self.drowsy_counter = 0
                    text = "DRIVER IS ALERT"
                    cv2.putText(image, text, (40, 40),
                            cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

                # # Play sound if activated
                if self.play_sound_alert:
                    if not self.playing:
                        self.play = self.wave_obj.play()
                    self.play_sound_alert=False
                    self.playing = self.play.is_playing()


            except Exception as e:
                print(e)
                text = 'WARNING! DRIVER NOT FOUND!'
                print(text)
                cv2.putText(image, text, (40, 40),
                            cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
            return image


        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            image = frame.to_ndarray(format="rgb24")
            annotated_image = self.draw_and_predict(image)
            return av.VideoFrame.from_ndarray(annotated_image, format="rgb24")

    webrtc_ctx = webrtc_streamer(
        key="drowsiness-detection",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=DrowsinessPredictor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

def about():
    st.write('Welcome to our drowiness detection system')
    st.markdown("""

     **About our app**
    - We are attempting to reduce the prevalence of car accidents caused by driver drowsiness.

    - Our app detects from a live webcam whether a driver is drowsy and if so, alerts them.

    - We consider a driver as drowsy if they are yawning or they have their eyes closed.

    **Examples**

    """)


    alert_image = Image.open('alert_example.png')
    st.image(alert_image, caption='Alert driver')

    drowsy_image = Image.open('drowsy_example.png')
    st.image(drowsy_image, caption='Drowsy driver')

############################ Sidebar + launching #################################################

object_detection_page = "F.R.I.D.A (Fatigue Related Intelligent Driver Assistant)"
about_page = "About F.R.I.D.A"

app_mode = st.sidebar.selectbox(
    "Choose the app mode",
    [
        object_detection_page,
        about_page
    ],
)

st.subheader(app_mode)

if app_mode == object_detection_page:
    app_drowsiness_detection()
if app_mode == about_page:
    about()
