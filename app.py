import streamlit as st 
import cv2
from deepface import DeepFace
import numpy as np

# Emotion mapping to YouTube search query
def get_youtube_search_query(emotion):
    emotion_to_song = {
        'happy': 'happy music playlist',
        'sad': 'sad music playlist',
        'angry': 'angry music playlist',
        'fear': 'calm music playlist',
        'surprise': 'surprise music playlist',
        'disgust': 'chill music playlist',
        'neutral': 'relaxing music playlist'
    }
    return emotion_to_song.get(emotion, 'relaxing music playlist')

# Function to process webcam input and detect emotion using DeepFace
def detect_emotion_from_webcam():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Error: Could not access the webcam.")
        return None, None

    emotion = None
    frame = None
    attempts = 0
    max_attempts = 10

    with st.spinner("Capturing and analyzing emotion..."):
        while attempts < max_attempts:
            ret, frame = cap.read()
            if not ret:
                attempts += 1
                continue

            try:
                analysis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=True)
                detected_emotion = analysis[0]['dominant_emotion']
                facial_area = analysis[0]['region']

                if detected_emotion:
                    emotion = detected_emotion
                    x, y, w, h = facial_area['x'], facial_area['y'], facial_area['w'], facial_area['h']
                    cv2.putText(frame, detected_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    break
            except Exception as e:
                st.warning(f"Error in emotion detection: {e}")
                attempts += 1

    cap.release()
    cv2.destroyAllWindows()

    if frame is None:
        st.error("Failed to capture a valid frame.")
        return None, None

    return emotion, frame

# Streamlit App UI
st.title('🎧 Emotion-Based Music Recommender')
st.markdown("""
Let your face do the talking, and let the music match the mood.  
Click below to detect your emotion and get a matching YouTube playlist 🎶
""")

# Button to start emotion detection
if st.button('🧠 Detect Emotion and Suggest Music'):
    detected_emotion, frame = detect_emotion_from_webcam()

    if detected_emotion and frame is not None:
        st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_column_width=True)
        st.success(f"**Detected Emotion:** {detected_emotion}")

        # Get YouTube URL
        search_query = get_youtube_search_query(detected_emotion)
        youtube_url = f"https://www.youtube.com/results?search_query={search_query.replace(' ', '+')}"

        # Display a clickable link (user must click)
        st.markdown(f"""
        ### 👉 [Click here to open YouTube Playlist for your mood 🎵]({youtube_url})""", unsafe_allow_html=True)
    else:
        st.error("No emotion detected. Please try again.")
