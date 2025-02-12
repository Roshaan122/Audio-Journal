import streamlit as st
import whisper
import sounddevice as sd
import numpy as np
import wave
import os
import json
import google.generativeai as genai
from datetime import date, datetime, timedelta
import plotly.graph_objects as go
from dotenv import load_dotenv

sd.default.device = 1 
# Load environment variables
load_dotenv()

# Constants
JOURNAL_FILE = "journal.json"
MOOD_FILE = "mood_ratings.json"

# Helper Functions
def load_journal():
    return json.load(open(JOURNAL_FILE, "r")) if os.path.exists(JOURNAL_FILE) else {}

def save_journal(journal):
    with open(JOURNAL_FILE, "w") as f:
        json.dump(journal, f, indent=4)

def load_mood_ratings():
    return json.load(open(MOOD_FILE, "r")) if os.path.exists(MOOD_FILE) else {}

def save_mood_rating(date, rating):
    mood_data = load_mood_ratings()
    mood_data[date] = rating
    with open(MOOD_FILE, "w") as f:
        json.dump(mood_data, f, indent=4)

def record_audio(duration=5, samplerate=44100, filename="temp.wav"):
    audio_data = sd.rec(int(samplerate * duration), samplerate=samplerate, channels=1, dtype=np.int16)
    sd.wait()
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(samplerate)
        wf.writeframes(audio_data.tobytes())
    return filename

def transcribe_audio(filename):
    return st.session_state.model.transcribe(filename)["text"]

def create_mood_graph(dates, mood_values, title, color):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates,
        y=mood_values,
        mode='lines+markers',
        name='Mood',
        line=dict(color=color, width=2),
        marker=dict(size=8)
    ))
    fig.update_layout(
        title=title,
        xaxis_title='Date',
        yaxis_title='Mood Rating (1-10)',
        yaxis_range=[0, 10],
        hovermode='x'
    )
    return fig

def initialize_models():
    if "model" not in st.session_state:
        st.session_state.model = whisper.load_model("tiny.en")
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        st.session_state.gen_model = genai.GenerativeModel("gemini-1.5-flash-latest")

def render_sidebar():
    with st.sidebar:
        st.header("💾 Backup & Restore")
        
        # Export section
        st.subheader("Export Data")
        if st.button("Export Journal"):
            journal_data = load_journal()
            st.download_button(
                label="Download Journal JSON",
                data=json.dumps(journal_data, indent=4),
                file_name="journal_backup.json",
                mime="application/json"
            )
        
        if st.button("Export Mood Ratings"):
            mood_data = load_mood_ratings()
            st.download_button(
                label="Download Mood Ratings JSON",
                data=json.dumps(mood_data, indent=4),
                file_name="mood_ratings_backup.json",
                mime="application/json"
            )

        # Import section
        st.subheader("Import Data")
        handle_file_upload("Journal", JOURNAL_FILE)
        handle_file_upload("Mood Ratings", MOOD_FILE)

def handle_file_upload(data_type, file_path):
    uploaded_file = st.file_uploader(f"Upload {data_type} JSON", type=['json'], key=f"{data_type.lower()}_upload")
    if uploaded_file is not None:
        try:
            imported_data = json.load(uploaded_file)
            if st.button(f"Import {data_type}"):
                with open(file_path, "w") as f:
                    json.dump(imported_data, f, indent=4)
                st.success(f"✅ {data_type} imported successfully!")
        except json.JSONDecodeError:
            st.error("Invalid JSON file")

def render_journal_column(journal_col):
    with journal_col:
        st.subheader("✍️ Journal Entry")
        selected_date = st.date_input("Select a date:", date.today())
        selected_date_str = str(selected_date)
        journal_data = load_journal()
        journal_entry = journal_data.get(selected_date_str, "")

        if st.button("🎙️ Record Journal Entry"):
            audio_file = record_audio()
            st.success("Recording completed!")
            journal_entry = transcribe_audio(audio_file)
            st.write("Transcription:", journal_entry)
            journal_data[selected_date_str] = journal_entry
            save_journal(journal_data)
            st.success("✅ Entry saved successfully!")

        new_entry = st.text_area("Write your journal entry:", journal_entry, height=200)

        if st.button("Save Entry ✍️"):
            journal_data[selected_date_str] = new_entry
            save_journal(journal_data)
            st.success("✅ Entry saved successfully!")
        
        return new_entry, selected_date_str

def render_mood_column(mood_col, new_entry, selected_date_str):
    with mood_col:
        st.subheader("📊 Mood Analysis")
        if st.button("Analyze Mood"):
            if new_entry:
                response = st.session_state.gen_model.generate_content(
                    f"Analyze this journal entry and rate the mood on a scale from 1 to 10. Return only the number:\n{new_entry}"
                )
                save_mood_rating(selected_date_str, response.text)
                st.write(f"Mood rating: {response.text}/10")
            else:
                st.warning("No journal entry to analyze")

        render_mood_trends()

def render_mood_trends():
    mood_ratings = load_mood_ratings()
    today = date.today()

    # Weekly Trend
    start_of_week = today - timedelta(days=today.weekday())
    week_dates = [(start_of_week + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(7)]
    week_moods = [float(mood_ratings.get(d, 0)) if d in mood_ratings else None for d in week_dates]
    weekly_fig = create_mood_graph(week_dates, week_moods, 'Weekly Mood Trend', '#1f77b4')
    st.plotly_chart(weekly_fig, use_container_width=True)

    # Monthly Trend
    start_of_month = today.replace(day=1)
    next_month = (start_of_month + timedelta(days=32)).replace(day=1)
    month_dates = [(start_of_month + timedelta(days=i)).strftime("%Y-%m-%d") 
                   for i in range((next_month - start_of_month).days)]
    month_moods = [float(mood_ratings.get(d, 0)) if d in mood_ratings else None for d in month_dates]
    monthly_fig = create_mood_graph(month_dates, month_moods, 
                                  f'Monthly Mood Trend - {today.strftime("%B %Y")}', '#2ecc71')
    st.plotly_chart(monthly_fig, use_container_width=True)

def render_past_entries(past_col):
    with past_col:
        st.subheader("📅 Past Entries")
        journal_data = load_journal()
        mood_ratings = load_mood_ratings()
        for entry_date, text in sorted(journal_data.items(), reverse=True):
            with st.expander(f"📅 {entry_date}"):
                st.write(text)
                if entry_date in mood_ratings:
                    st.write(f"Mood: {mood_ratings[entry_date]}/10")

def main():
    st.set_page_config(layout="wide", initial_sidebar_state="expanded")
    st.title("📖 Daily Journal")
    
    initialize_models()
    render_sidebar()

    # Create three columns
    journal_col, mood_col, past_col = st.columns([1.2, 1.2, 0.8])
    
    # Render columns
    new_entry, selected_date_str = render_journal_column(journal_col)
    render_mood_column(mood_col, new_entry, selected_date_str)
    render_past_entries(past_col)

if __name__ == "__main__":
    main()