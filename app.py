import streamlit as st
import requests
import tempfile
import os
import base64
from playsound import playsound
import json

def text_2_speech(input1):
    url = "https://api.sarvam.ai/text-to-speech"
    payload = {
        "inputs": [f"{input1}"],
        "target_language_code": "hi-IN",
        "speaker": "meera",
        "pitch": 0,
        "pace": 1.2,
        "loudness": 1.5,
        "speech_sample_rate": 8000,
        "enable_preprocessing": True,
        "model": "bulbul:v1"
    }
    headers = {
        "api-subscription-key": "3663dd5f-7cc8-49d3-bac3-b5a7dd3d591e",
        "Content-Type": "application/json"
    }

    response = requests.post(url, json=payload, headers=headers)
    if response.status_code == 200:
        response_data = response.json()
        audio_base64 = response_data['audios'][0]
        audio_data = base64.b64decode(audio_base64)
        with open("output_audio.wav", "wb") as audio_file:
            audio_file.write(audio_data)
    playsound("output_audio.wav")

def read_and_save_file():
    global file_path
    st.session_state["messages"] = []
    st.session_state["user_input"] = ""

    for file in st.session_state["file_uploader"]:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tf:
            tf.write(file.getbuffer())
            file_path = tf.name

        # Send file path to the ingest endpoint
        ingest_response = requests.post(f"{API_URL}/ingest", json={"pdf_file_path": file_path})
        if ingest_response.status_code != 200:
            st.error("Error loading PDF for querying.")

st.set_page_config(page_title="RAG System", page_icon="📚")

API_URL = "http://localhost:8000"

if "messages" not in st.session_state:
    st.session_state["messages"] = []

st.title("Chat with RAG System")
st.subheader("Upload a document")
st.file_uploader(
    "Upload document",
    type=["pdf"],
    key="file_uploader",
    on_change=read_and_save_file,
    label_visibility="collapsed",
    accept_multiple_files=True,
)

st.session_state["ingestion_spinner"] = st.empty()
user_input = st.text_input("Enter your question here:", key="input")

if st.button("Ask"):
    if user_input:
        with st.spinner("Thinking..."):

            response = requests.post(f"{API_URL}/query/", json={"question": user_input})
            if response.status_code == 200:
                result = response.json()
                #print(result_str)
                # result_str = result["response"]
                # text_2_speech(result_str)
                # st.audio("output_audio.wav", format='audio/wav')
                answer = result["response"]
                st.session_state["messages"].append(("User", user_input))
                st.session_state["messages"].append(("Assistant", answer))
                # os.remove("output_audio.wav")
            else:
                st.error("Error in querying the RAG system.")

if st.session_state["messages"]:
    st.write("---")
    for sender, msg in st.session_state["messages"]:
        if sender == "User":
            st.markdown(
                f"""
                <div style='background-color: #fff; color: #003399; padding: 10px; border-radius: 10px; margin: 5px 0; text-align: right;'>
                    <strong>User:</strong> {msg}
                </div>
                """, 
                unsafe_allow_html=True
        )
        else:
            st.markdown(
                f"""
                <div style='background-color: #fff; color: #003399; padding: 10px; border-radius: 10px; margin: 5px 0; text-align: left;'>
                    <strong>Assistant:</strong> {msg}
                </div>
                """, 
                unsafe_allow_html=True
            )




