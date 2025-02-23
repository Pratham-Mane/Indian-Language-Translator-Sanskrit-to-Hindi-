import tkinter as tk
from tkinter import messagebox
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import speech_recognition as sr
import os
from gtts import gTTS
import json
import pyttsx3

# Load tokenizers and model
def load_tokenizer(tokenizer_file):
    with open(tokenizer_file, 'r', encoding='utf-8') as f:
        return tf.keras.preprocessing.text.tokenizer_from_json(f.read())

sanskrit_tokenizer = load_tokenizer('fine_tuned_sanskrit_tokenizer3.json')
hindi_tokenizer = load_tokenizer('fine_tuned_hindi_tokenizer3.json')
model = tf.keras.models.load_model('fine_tuned_sanskrit_hindi_model3.keras')

# Define max sequence lengths
max_len_sanskrit = 50
max_len_hindi = 50

# Function to perform translation using the custom model
def translate(input_sentence, source_lang):
    if source_lang == "sa":
        tokenizer_input = sanskrit_tokenizer
        tokenizer_output = hindi_tokenizer
        max_len_input = max_len_sanskrit
        max_len_output = max_len_hindi
    else:
        tokenizer_input = hindi_tokenizer
        tokenizer_output = sanskrit_tokenizer
        max_len_input = max_len_hindi
        max_len_output = max_len_sanskrit
    
    input_seq = tokenizer_input.texts_to_sequences([input_sentence])
    input_seq = pad_sequences(input_seq, maxlen=max_len_input, padding='post')

    decoder_seq = np.zeros((1, max_len_output))
    predictions = model.predict([input_seq, decoder_seq])

    predicted_sentence = np.argmax(predictions, axis=-1)
    predicted_words = [tokenizer_output.index_word[idx] for idx in predicted_sentence[0] if idx > 0]

    return ' '.join(predicted_words)

# Function to handle text translation
def translate_text():
    source_text = input_text.get("1.0", "end").strip()
    if not source_text:
        messagebox.showwarning("Input Error", "Please provide input text.")
        return
    
    source_lang = "sa" if mode_var.get() == "1" else "hi"
    try:
        translated = translate(source_text, source_lang)
        output_text.delete("1.0", "end")
        output_text.insert("1.0", translated)
    except Exception as e:
        messagebox.showerror("Translation Error", f"Error occurred: {e}")

# Function to handle speech recognition
def recognize_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        try:
            messagebox.showinfo("Listening", "Please speak something...")
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
            text = recognizer.recognize_google(audio, language="hi-IN")
            input_text.delete("1.0", "end")
            input_text.insert("1.0", text)
        except sr.UnknownValueError:
            messagebox.showerror("Speech Recognition Error", "Sorry, could not understand the audio.")
        except sr.RequestError as e:
            messagebox.showerror("Speech Recognition Error", f"Error: {e}")

# Function to play text-to-speech
def play_audio():
    translated_text = output_text.get("1.0", "end").strip()
    if not translated_text:
        messagebox.showwarning("Output Error", "No translated text to play.")
        return
    try:
        engine = pyttsx3.init()
        engine.say(translated_text)
        engine.runAndWait()
    except Exception as e:
        messagebox.showerror("Audio Error", f"Error occurred: {e}")

# Tkinter GUI
root = tk.Tk()
root.title("Sanskrit-Hindi Translator")

# Input Frame
input_frame = tk.LabelFrame(root, text="Input Text", padx=10, pady=10)
input_frame.pack(padx=10, pady=5, fill="both", expand="yes")

input_text = tk.Text(input_frame, height=5, wrap="word")
input_text.pack(fill="both", expand="yes")

speech_button = tk.Button(input_frame, text="🎤 Speak", command=recognize_speech)
speech_button.pack(pady=5)

# Mode Selection
mode_frame = tk.LabelFrame(root, text="Translation Mode", padx=10, pady=10)
mode_frame.pack(padx=10, pady=5, fill="both", expand="yes")

mode_var = tk.StringVar(value="1")
tk.Radiobutton(mode_frame, text="Sanskrit to Hindi", variable=mode_var, value="1").pack(anchor="w")
tk.Radiobutton(mode_frame, text="Hindi to Sanskrit", variable=mode_var, value="2").pack(anchor="w")

# Output Frame
output_frame = tk.LabelFrame(root, text="Translated Text", padx=10, pady=10)
output_frame.pack(padx=10, pady=5, fill="both", expand="yes")

output_text = tk.Text(output_frame, height=5, wrap="word")
output_text.pack(fill="both", expand="yes")

# Action Buttons
button_frame = tk.Frame(root)
button_frame.pack(pady=10)

translate_button = tk.Button(button_frame, text="Translate", command=translate_text)
translate_button.grid(row=0, column=0, padx=10)

play_button = tk.Button(button_frame, text="🔊 Play Audio", command=play_audio)
play_button.grid(row=0, column=1, padx=10)

# Run the Tkinter main loop
root.mainloop()
