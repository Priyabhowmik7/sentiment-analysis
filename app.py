import os
import re
from flask import Flask, request, render_template
import speech_recognition as sr
import pyttsx3
from nltk.sentiment.vader import SentimentIntensityAnalyzer

app = Flask(__name__)

def remove_special_characters_from_file(input_file):
    try:
        with open(input_file, 'r') as file:
            file_content = file.read()
        
        pattern = r'[^a-zA-Z\s]'
        cleaned_content = re.sub(pattern, '', file_content)
        return cleaned_content
    except FileNotFoundError:
        return None

def sentiment_analysis(text):
    sia = SentimentIntensityAnalyzer()
    return sia.polarity_scores(text)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' in request.files:
        file = request.files['file']
        if file.filename != '':
            content = file.read().decode('utf-8')
            cleaned_content = re.sub(r'[^a-zA-Z\s]', '', content)
            analysis = sentiment_analysis(cleaned_content)
            return render_template('index.html', analysis=analysis)

    elif 'voice' in request.form:
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source)

            try:
                text = recognizer.recognize_google(audio)
                analysis = sentiment_analysis(text)
                return render_template('index.html', analysis=analysis, text=text)
            except sr.UnknownValueError:
                return render_template('index.html', error="Sorry, I couldn't understand your voice.")
            except sr.RequestError as e:
                return render_template('index.html', error=f"Request error: {e}")
    
    return render_template('index.html', error="No input provided.")

if __name__ == "__main__":
    app.run(debug=True)
