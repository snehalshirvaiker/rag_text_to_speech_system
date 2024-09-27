import streamlit as st
import requests
import requests
import os
import base64
from playsound import playsound

def text_2_speech(input):
    url = "https://api.sarvam.ai/text-to-speech"
    payload = {
        "inputs": [f"{input}"],
        "target_language_code": "hi-IN",
        "speaker": "meera",
        "pitch": 0,
        "pace": 1.65,
        "loudness": 1.5,
        "speech_sample_rate": 8000,
        "enable_preprocessing": True,
        "model": "bulbul:v1"
    }
    headers = {
        "api-subscription-key": "0d3191dc-cbaf-429d-8693-6ce94c622901",
        "Content-Type": "application/json"
    }

    response = requests.post(url, json=payload, headers=headers)
    if response.status_code == 200:
        audio_file_path = os.path.join(os.getcwd(), "output.mp3")  # Use absolute path
        response_data = response.json()
        audio_base64 = response_data['audios'][0]
        audio_data = base64.b64decode(audio_base64)
        with open("output_audio.wav", "wb") as audio_file:
            audio_file.write(audio_data)
    playsound("output_audio.wav")


st.set_page_config(page_title="RAG NCERT System", page_icon="📚")

# FastAPI URL
API_URL = "http://localhost:8000"

# ChatPDF Interaction State
if "messages" not in st.session_state:
    st.session_state["messages"] = []

st.title("NCERT Chat with RAG System")
st.subheader("Ask a question about the content")

user_input = st.text_input("Enter your question here:", key="input")

if st.button("Ask"):
    if user_input:
        with st.spinner("Thinking..."):
            # Send the user input to the FastAPI endpoint
            response = requests.post(f"{API_URL}/query/", json={"question": user_input})
            if response.status_code == 200:
                result = response.json()
                print(result)
                text_2_speech(result)
                st.audio("output_audio.wav", format='audio/wav')
                answer = result["response"]
                st.session_state["messages"].append(("User", user_input))
                st.session_state["messages"].append(("Assistant", answer))
                os.remove("output_audio.wav")
            else:
                st.error("Error in querying the RAG system.")

# Display messages in chat format
if st.session_state["messages"]:
    st.write("---")
    for sender, msg in st.session_state["messages"]:
        if sender == "User":
            st.markdown(f"<div style='text-align: right; color: blue;'>**User:** {msg}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div style='text-align: left; color: green;'>**Assistant:** {msg}</div>", unsafe_allow_html=True)
