# Import for streamlit
import streamlit as st
import av
import time
import simpleaudio as sa
from streamlit_webrtc import (
    RTCConfiguration,
    VideoProcessorBase,
    WebRtcMode,
    webrtc_streamer,
)

#Import for handling image
from cv2 import cv2

# Models and preprocessing
from project_drowsy.predict import make_prediction

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

#Main intelligence of the file, class to launch a webcam, detect faces, then detect drowsiness and output probability for drowsiness
def app_drowsiness_detection():
    class DrowsinessPredictor(VideoProcessorBase):

        def __init__(self) -> None:
            self.drowsy_counter = 0
            self.drowsy_flag = False
            pass

        def draw_and_predict(self, image):

            preprocess_params = dict(webcam=image,
                                    image_size=145,
                                    predict=True)

            try:
                if self.drowsy_flag:
                    wave_obj = sa.WaveObject.from_wave_file('airhorn.wav')
                    play_obj = wave_obj.play()
                    play_obj.wait_done()
                    time.sleep(1) # Sleep for 3 seconds
                    self.drowsy_flag=False



                prediction, probs, face_coords, left_eye_coords, right_eye_coords  = make_prediction(**preprocess_params)

                # draw eye bounding boxes using co-ordinates of the bounding box (from preprocessing)
                xmin_l,xmax_l,ymin_l,ymax_l = left_eye_coords
                xmin_r, xmax_r, ymin_r, ymax_r = right_eye_coords
                #left eye
                cv2.rectangle(image, (xmax_l,ymax_l), (xmin_l,ymin_l),
                            color=(0, 255, 0), thickness=2)
                #right eye
                cv2.rectangle(image, (xmax_r, ymax_r), (xmin_r, ymin_r),
                            color=(0, 255, 0),
                            thickness=2)

                #draw face box
                #print(face_coords)
                xmin, xmax, ymin, ymax = face_coords
                cv2.rectangle(image, (xmax, ymax), (xmin, ymin),
                            color=(0, 255, 0),
                            thickness=3)

                # evaluate
                print(prediction)
                print(probs)
                # # Put text on image


                if ("closed" in prediction) or ("yawn" in prediction):
                    self.drowsy_counter += 1
                    if self.drowsy_counter >= 5:
                        self.drowsy_flag=True
                        text = "Prediction = Drowsy"
                        colour = (255, 0, 0)
                    else:
                        text = "Prediction = Alert"
                        colour = (0, 255, 0)
                    cv2.putText(image, text, (40, 40),
                            cv2.FONT_HERSHEY_PLAIN, 2, colour, 2)
                else:
                    self.drowsy_counter = 0
                    text = "Prediction = Alert"
                    cv2.putText(image, text, (40, 40),
                            cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
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


############################ Sidebar + launching #################################################

object_detection_page = "Drowsiness video detector"

app_mode = st.sidebar.selectbox(
    "Choose the app mode",
    [
        object_detection_page,
    ],
)
st.subheader(app_mode)
if app_mode == object_detection_page:
    app_drowsiness_detection()
