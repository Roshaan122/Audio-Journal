import streamlit as st
import whisper
import sounddevice as sd
import numpy as np
import wave
import os
import json
import google.generativeai as genai
from datetime import date

# Initialize Whisper model and configure Gemini API if not already in session state
if "model" not in st.session_state:
    st.session_state.model = whisper.load_model("tiny.en")
    genai.configure(api_key="AIzaSyBNo2lKZ-eKWPuO4zr9h_3DyYnu8ub8ir4")  # Ensure to set the correct API key here
    st.session_state.gen_model = genai.GenerativeModel("gemini-1.5-flash-latest")

# File to store journal entries
JOURNAL_FILE = "journal.json"

# Record audio function
def record_audio(duration=5, samplerate=44100, filename="temp.wav"):
    st.write(f"Recording for {duration} seconds...")
    audio_data = sd.rec(int(samplerate * duration), samplerate=samplerate, channels=1, dtype=np.int16)
    sd.wait()
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit PCM
        wf.setframerate(samplerate)
        wf.writeframes(audio_data.tobytes())
    return filename

# Transcribe audio using Whisper
def transcribe_audio(filename):
    model = st.session_state.model
    result = model.transcribe(filename)
    return result["text"]

# Load existing journal data
def load_journal():
    if os.path.exists(JOURNAL_FILE):
        with open(JOURNAL_FILE, "r") as f:
            return json.load(f)
    return {}

# Save journal data to a file
def save_journal(journal):
    with open(JOURNAL_FILE, "w") as f:
        json.dump(journal, f, indent=4)

# Initialize journal
journal_data = load_journal()

# Streamlit UI
st.title("📖 Daily Journal")

# Date picker
selected_date = st.date_input("Select a date:", date.today())
selected_date_str = str(selected_date)

# Load existing entry for the selected date
journal_entry = journal_data.get(selected_date_str, "")

# Text area for journal entry
journal_entry1 = ""  # Variable to hold transcribed text

if st.button("🎙️ Record Journal Entry"):
    audio_file = record_audio()
    st.success("Recording completed!")
    
    st.write("Transcribing...")
    journal_entry1 = transcribe_audio(audio_file)  # Capture transcribed text
    
    # Append the transcribed text to the journal entry
    journal_entry += " " + journal_entry1
    st.write("Transcription:", journal_entry1)  # Show transcribed text
    new_entry = st.text_area("Write your journal entry:", journal_entry, height=200)
    journal_data[selected_date_str] = new_entry  # Save the updated entry
    save_journal(journal_data)  # Save to JSON file
    st.success("✅ Entry saved successfully!")
else:
    new_entry = st.text_area("Write your journal entry:", journal_entry, height=200)

# Save button
if st.button("Save Entry ✍️"):
    journal_data[selected_date_str] = new_entry  # Save the updated entry
    save_journal(journal_data)  # Save to JSON file
    st.success("✅ Entry saved successfully!")

# Display previous entries
st.subheader("📅 Past Entries")
for entry_date, text in sorted(journal_data.items(), reverse=True):
    with st.expander(f"📅 {entry_date}"):
        st.write(text)

# Analyze mood button
if st.button("Analyze Mood"):
    # Prepare the journal content as a summary for mood analysis
    journal_summary = " ".join(journal_data.values())[:1000]  # Limit input size to 1000 characters for API
    response = st.session_state.gen_model.generate_content(
        f"Analyze the following journal entries and rate the mood of the person on a scale from 1 to 10. "
        f"Return only the number:\n{journal_summary}"
    )
    st.write(f"Mood rating (1-10): {response.text}")
