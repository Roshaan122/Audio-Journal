import streamlit as st
import whisper
import os
import json
import google.generativeai as genai
from datetime import date, datetime, timedelta
import plotly.graph_objects as go
from dotenv import load_dotenv
from audiorecorder import audiorecorder

# Load environment variables
load_dotenv()

def initialize_session_state():
    if 'journal_entries' not in st.session_state:
        st.session_state.journal_entries = {}
    if 'mood_ratings' not in st.session_state:
        st.session_state.mood_ratings = {}

def record_audio(filename="temp.wav"):
    audio = audiorecorder("🎙️ Click to record", "⏹️ Click to stop recording")
    if len(audio) > 0:
        audio.export(filename, format="wav")
        st.audio(audio.export().read())
        st.write(f"Duration: {audio.duration_seconds:.2f} seconds")
        return filename
    return None

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
            st.download_button(
                label="Download Journal JSON",
                data=json.dumps(st.session_state.journal_entries, indent=4),
                file_name="journal_backup.json",
                mime="application/json"
            )
        
        if st.button("Export Mood Ratings"):
            st.download_button(
                label="Download Mood Ratings JSON",
                data=json.dumps(st.session_state.mood_ratings, indent=4),
                file_name="mood_ratings_backup.json",
                mime="application/json"
            )

        # Import section
        st.subheader("Import Data")
        journal_file = st.file_uploader("Upload Journal JSON", type=['json'], key="journal_upload")
        if journal_file is not None:
            try:
                st.session_state.journal_entries = json.load(journal_file)
                st.success("✅ Journal imported successfully!")
            except json.JSONDecodeError:
                st.error("Invalid JSON file")

        mood_file = st.file_uploader("Upload Mood Ratings JSON", type=['json'], key="mood_upload")
        if mood_file is not None:
            try:
                st.session_state.mood_ratings = json.load(mood_file)
                st.success("✅ Mood ratings imported successfully!")
            except json.JSONDecodeError:
                st.error("Invalid JSON file")

def render_journal_column(journal_col):
    with journal_col:
        st.subheader("✍️ Journal Entry")
        selected_date = st.date_input("Select a date:", date.today())
        selected_date_str = str(selected_date)
        journal_entry = st.session_state.journal_entries.get(selected_date_str, "")

        # Audio recording section
        audio_file = record_audio()
        if audio_file:
            journal_entry = transcribe_audio(audio_file)
            st.write("Transcription:", journal_entry)
            st.session_state.journal_entries[selected_date_str] = journal_entry
            st.success("✅ Entry saved successfully!")

        new_entry = st.text_area("Write your journal entry:", journal_entry, height=200)

        if st.button("Save Entry ✍️"):
            st.session_state.journal_entries[selected_date_str] = new_entry
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
                st.session_state.mood_ratings[selected_date_str] = response.text
                st.write(f"Mood rating: {response.text}/10")
            else:
                st.warning("No journal entry to analyze")

        render_mood_trends()

def render_mood_trends():
    today = date.today()

    # Weekly Trend
    start_of_week = today - timedelta(days=today.weekday())
    week_dates = [(start_of_week + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(7)]
    week_moods = [float(st.session_state.mood_ratings.get(d, 0)) if d in st.session_state.mood_ratings else None for d in week_dates]
    weekly_fig = create_mood_graph(week_dates, week_moods, 'Weekly Mood Trend', '#1f77b4')
    st.plotly_chart(weekly_fig, use_container_width=True)

    # Monthly Trend
    start_of_month = today.replace(day=1)
    next_month = (start_of_month + timedelta(days=32)).replace(day=1)
    month_dates = [(start_of_month + timedelta(days=i)).strftime("%Y-%m-%d") 
                   for i in range((next_month - start_of_month).days)]
    month_moods = [float(st.session_state.mood_ratings.get(d, 0)) if d in st.session_state.mood_ratings else None for d in month_dates]
    monthly_fig = create_mood_graph(month_dates, month_moods, 
                                  f'Monthly Mood Trend - {today.strftime("%B %Y")}', '#2ecc71')
    st.plotly_chart(monthly_fig, use_container_width=True)

def render_past_entries(past_col):
    with past_col:
        st.subheader("📅 Past Entries")
        for entry_date, text in sorted(st.session_state.journal_entries.items(), reverse=True):
            with st.expander(f"📅 {entry_date}"):
                st.write(text)
                if entry_date in st.session_state.mood_ratings:
                    st.write(f"Mood: {st.session_state.mood_ratings[entry_date]}/10")

def main():
    st.set_page_config(layout="wide", initial_sidebar_state="expanded")
    st.title("📖 Daily Journal")
    
    initialize_session_state()
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