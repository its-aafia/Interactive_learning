import streamlit as st
from PIL import Image
import base64
import pyttsx3
import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
import google.generativeai as genai
import threading
import random

# Streamlit setup
st.set_page_config(layout="centered", page_title="Interactive Learning App")

# Initialize voice engine
engine = pyttsx3.init()

# Function to encode image as base64 for use in CSS
def get_base64(file_path):
    with open(file_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Function to speak text in a funny, engaging voice
def speak_text(text, name):
    engine.setProperty('rate', 150)  # Set speech rate (words per minute)
    engine.setProperty('volume', 1)  # Set volume (0.0 to 1.0)
    engine.setProperty('pitch', random.randint(50, 150))  # Set random pitch for a playful tone
    engine.say(f"Hey {name}, {text}")  # Include the child's name in the speech
    engine.runAndWait()

# Step 1: Input Screen
if "step" not in st.session_state:
    st.session_state.step = "input_screen"
    st.session_state.name = ""
    st.session_state.age = 0
    st.session_state.voice_enabled = True  # Voice option enabled by default

if st.session_state.step == "input_screen":
    # Load the banner image
    image_path = r"A:\SELF course\PROJECTS\Interactive_learning\Interactive_learning\Kids.png"  # Update with your image file path
    encoded_image = get_base64(image_path)

    # Add CSS for the background and input styling
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded_image}");
            background-size: cover;
            background-repeat: no-repeat;
            background-position: center;
            height: 100vh;
            width: 100vw;
        }}
        .input-container {{
            position: absolute;
            top: 45%;
            left: 50%;
            transform: translate(-50%, -50%);
            text-align: center;
            color: white;
            background-color: rgba(0, 0, 0, 0.6);
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0px 4px 15px rgba(0, 0, 0, 0.3);
            width: 80%;
            max-width: 500px;
        }}
        .input-container h2 {{
            font-size: 24px;
            margin-bottom: 20px;
        }}
        input, select, button {{
            font-size: 18px;
            padding: 10px;
            margin: 10px;
            border-radius: 5px;
            border: none;
            outline: none;
        }}
        button {{
            background-color: #6200EE;
            color: white;
            cursor: pointer;
        }}
        button:hover {{
            background-color: #3700B3;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Input container for name and age
    st.markdown(
        """
        <div class="input-container">
            <h2>Welcome to the Interactive Learning App!</h2>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.session_state.name = st.text_input(
            "Enter Your Name",
            placeholder="Type your name...",
            key="name_input",
        )
        st.session_state.age = st.number_input(
            "Enter Your Age",
            min_value=1,
            max_value=100,
            value=10,
            step=1,
            key="age_input",
        )
        st.session_state.voice_enabled = st.checkbox("Enable Voice", value=True)

        if st.button("Let's Begin!", key="start_button"):
            st.session_state.step = "welcome_screen"

# Step 2: Welcome Screen
if st.session_state.step == "welcome_screen":
    # Display welcome message
    st.markdown(
        f"<h1 style='text-align:center; color:purple;'>✨ Welcome, {st.session_state.name}! ✨</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<h2 style='text-align:center; color:blue;'>Get ready to explore and learn! 🎉</h2>",
        unsafe_allow_html=True,
    )

    # Voice output in a separate thread
    if st.session_state.voice_enabled:
        threading.Thread(target=speak_text, args=(f"Welcome {st.session_state.name}. Get ready to explore and learn!", st.session_state.name)).start()

    if st.button("Next", key="next_button"):
        st.session_state.step = "main_app"

# Step 3: Main Application (App Code)
if st.session_state.step == "main_app":
    st.write("Come , Draw and get answer and learn with Fun")

    # App code
    col1, col2 = st.columns([3, 2])
    with col1:
        run = st.checkbox('Run', value=True)
        FRAME_WINDOW = st.image([])
    with col2:
        st.title("Answer")
        output_text_area = st.empty()

    # Configure the Gemini AI API
    genai.configure(api_key="AIzaSyBCLUuMgOs3rUrLhrp4Sn6TD-mb1qCxOt8")
    model = genai.GenerativeModel("gemini-1.5-flash")

    # Initialize webcam
    cap = cv2.VideoCapture(0)  # Use external webcam (index 1)
    cap.set(3, 1920)  # Set width to 1920
    cap.set(4, 1080)  # Set height to 1080

    # Initialize hand detector
    detector = HandDetector(maxHands=1)

    # Functions
    def getHandInfo(img):
        hands, img = detector.findHands(img, draw=False, flipType=True)
        if hands:
            hand = hands[0]
            lmList = hand["lmList"]
            fingers = detector.fingersUp(hand)
            return fingers, lmList
        return None

    def draw(info, prev_pos, canvas):
        fingers, lmList = info
        current_pos = None
        if fingers == [0, 1, 0, 0, 0]:  # Drawing gesture (index finger up)
            current_pos = lmList[8][0:2]
            if prev_pos is None:
                prev_pos = current_pos
            cv2.line(canvas, prev_pos, current_pos, (0, 255, 0), 15)
        elif fingers == [1, 0, 0, 0, 0]:  # Erase gesture (thumb up)
            canvas = np.zeros_like(canvas)
        return current_pos, canvas

    def sendToAI(model, canvas, fingers):
        if fingers == [0, 1, 1, 1, 0]:  # Trigger for AI response
            pil_image = Image.fromarray(canvas)
            prompt = """You are a friendly and humorous AI art teacher for kids aged 5 to 10. Respond to drawings enthusiastically, encourage creativity, and make learning fun and should be 2 to 10 lines only.

For Specific Drawings:

if Draw 'st': Share funny, common kids' stories in an engaging way.
if Draw 'Po': Recite funny and popular poems for kids.
if Draw 'AB': Start singing "A B C D E..." in a playful and amusing way.
if Draw Messy Drawing: "Hmm, this looks like a creative start! Maybe it's an abstract masterpiece? Keep experimenting!"
if Draw '111': Count from 1 to 20 in a funny, attention-grabbing manner.
if Draw Math Problems: Solve the problem in the best way 
            """
            response = model.generate_content([prompt, pil_image])
            return response.text
        return ""

    # Main loop
    prev_pos = None
    canvas = None
    while run:
        success, img = cap.read()
        if not success:
            st.warning("Failed to access the webcam.")
            break

        img = cv2.flip(img, 1)
        if canvas is None:
            canvas = np.zeros_like(img)

        info = getHandInfo(img)
        if info:
            fingers, lmList = info
            prev_pos, canvas = draw(info, prev_pos, canvas)
            output_text = sendToAI(model, canvas, fingers)
            if output_text:
                output_text_area.text(output_text)

                # Voice output for AI response in a separate thread
                if st.session_state.voice_enabled:
                    threading.Thread(target=speak_text, args=(f"{output_text} Hey {st.session_state.name}, let's keep going!", st.session_state.name)).start()

        # Combine canvas with original image
        image_combined = cv2.addWeighted(img, 0.7, canvas, 0.3, 0)
        FRAME_WINDOW.image(image_combined, channels="BGR")

    # Release resources
    cap.release()
    cv2.destroyAllWindows()