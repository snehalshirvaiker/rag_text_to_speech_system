import requests
import os
import base64
from playsound import playsound
url = "https://api.sarvam.ai/text-to-speech"
payload = {
    "inputs": ["Hello"],
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