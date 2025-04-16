from flask import Flask, request, jsonify, render_template, send_from_directory
from transformers import M2M100ForConditionalGeneration, MBartForConditionalGeneration
from transformers import M2M100Tokenizer, MBart50TokenizerFast
import torch
import requests
import os
from gtts import gTTS
import io
import wave
import speech_recognition as sr

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/audio'

# Create audio directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load translation models and tokenizers
sanskrit_to_english_model_name = "model_Sanskrit_English"
english_to_indian_model_name = "model_Indic_Lang"

# Initialize models and tokenizers
sanskrit_tokenizer = M2M100Tokenizer.from_pretrained(sanskrit_to_english_model_name)
sanskrit_model = M2M100ForConditionalGeneration.from_pretrained(sanskrit_to_english_model_name)
indian_tokenizer = MBart50TokenizerFast.from_pretrained(english_to_indian_model_name)
indian_model = MBartForConditionalGeneration.from_pretrained(english_to_indian_model_name)

# Language code mapping
language_codes = {
    'bn': 'bn_IN', 'gu': 'gu_IN', 'hi': 'hi_IN', 'mr': 'mr_IN',
    'ta': 'ta_IN', 'te': 'te_IN', 'ml': 'ml_IN', 'pa': 'pa_IN',
    'ne': 'ne_NP', 'si': 'si_LK', 'ur': 'ur_PK'
}

# Language names for text-to-speech
language_names = {
    'hi': 'hi', 'bn': 'bn', 'gu': 'gu', 'mr': 'mr',
    'ta': 'ta', 'te': 'te', 'ml': 'ml', 'pa': 'pa',
    'ne': 'ne', 'si': 'si', 'ur': 'ur'
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chatbot', methods=['GET', 'POST'])
def chatbot():
    return render_template('chatbot.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    api_key = request.form.get("api_key")
    return get_Chat_response(msg, api_key)

def get_Chat_response(text, api_key):
    botpress_url = ""
    headers = {'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json'}
    payload = {"type": "text", "text": text, "channel": "web", "userId": "user-id"}
    response = requests.post(botpress_url, headers=headers, json=payload)
    return response.json().get('responses', ['Sorry, I could not process your request.'])[0] if response.status_code == 200 else "Error: " + response.text

@app.route('/translate', methods=['POST'])
def translate():
    data = request.json
    text = data.get('text', '')
    target_language = data.get('target_language', 'hi')
    
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    try:
        # Sanskrit to English
        inputs = sanskrit_tokenizer(text, return_tensors="pt")
        outputs = sanskrit_model.generate(**inputs)
        english_translation = sanskrit_tokenizer.decode(outputs[0], skip_special_tokens=True)
        # print(english_translation)
        
        # English to Indian language
        indian_tokenizer.src_lang = "en_XX"
        encoded_english = indian_tokenizer(english_translation, return_tensors="pt")
        generated_tokens = indian_model.generate(
            **encoded_english,
            forced_bos_token_id=indian_tokenizer.lang_code_to_id[language_codes[target_language]]
        )
        translated_text = indian_tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
        
        return jsonify({
            'translated_text': translated_text,
            'target_language': target_language
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/speech-to-text', methods=['POST'])
def speech_to_text():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    
    try:
        audio_file = request.files['audio']
        recognizer = sr.Recognizer()
        
        # Convert to AudioFile format
        with io.BytesIO() as wav_io:
            with wave.open(wav_io, 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(16000)
                wav_file.writeframes(audio_file.read())
            
            wav_io.seek(0)
            
            with sr.AudioFile(wav_io) as source:
                audio_data = recognizer.record(source)
                
                # Recognize using Google Web Speech API with Sanskrit language hint
                text = recognizer.recognize_google(
                    audio_data,
                    language='sa-IN'  # Sanskrit language code for India
                )
                return jsonify({'text': text})
                
    except sr.UnknownValueError:
        return jsonify({'error': 'Speech recognition could not understand the Sanskrit audio'}), 400
    except sr.RequestError as e:
        return jsonify({'error': f'Could not request results from speech recognition service; {e}'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/text-to-speech', methods=['POST'])
def text_to_speech():
    data = request.json
    text = data.get('text', '')
    lang = data.get('lang', 'hi')
    
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    try:
        tts = gTTS(text=text, lang=language_names.get(lang, 'hi'))
        filename = f"output_{lang}.mp3"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        tts.save(filepath)
        return jsonify({'audio_url': f'/static/audio/{filename}'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/static/audio/<filename>')
def serve_audio(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run()