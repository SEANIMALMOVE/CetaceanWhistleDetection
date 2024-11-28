import gradio as gr
import librosa
import numpy as np
from PIL import Image
import shutil
import os
from io import BytesIO
import matplotlib.pyplot as plt
import tensorflow as tf # type: ignore
from marine_mammals_classes_mapping import CLASSES_MAPPING as classes_mapping

EXPERIMENT = "0_WatkinsMarineMammals_Base"

model = tf.keras.models.load_model(f"DeepLearning/BirdNET/Models/0_WatkinsMarineMammals_Base/")

# Function to convert audio clip to mel spectrogram image
def audio_to_mel_spectrogram(audio_clip):
    # Load the audio clip
    # audio_clip_path = process_file(audio_clip)

    y, sr = librosa.load(audio_clip, sr=None)

    # Define the frequency range
    fmin = 1  # Minimum frequency (0 Hz)
    fmax = 48000  # Maximum frequency (32000 Hz)

    fig, ax = plt.subplots(figsize=(12, 6))  # Set the background color to black
    D = librosa.amplitude_to_db(librosa.stft(y), ref=np.max)
    librosa.display.specshow(D, sr=sr, x_axis="time", y_axis="log", fmin=fmin, fmax=fmax, ax=ax)  # Specify frequency range
    ax.axis('off')  # Remove axes

    # Convert the plot to an image object
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, transparent=True)
    buf.seek(0)
    mel_spectrogram_image = Image.open(buf)

    return mel_spectrogram_image

def update_mel_spectrogram(audio_clip):
    mel_spectrogram_image = audio_to_mel_spectrogram(audio_clip)
    # Convert the image to PNG format before returning
    mel_spectrogram_image = mel_spectrogram_image.convert("RGB")
    return mel_spectrogram_image

# Function to preprocess audio
def preprocess_audio(file_path, target_sr=48000, duration=3):
    # Load audio
    signal, sr = librosa.load(file_path, sr=target_sr, mono=True)
    
    # Calculate the maximum length in samples
    max_length = target_sr * duration  # e.g., 48000 * 3
    
    # If the signal is shorter than the desired length, pad with silence
    if len(signal) < max_length:
        padding = max_length - len(signal)
        pad_left = padding // 2
        pad_right = padding - pad_left
        signal = np.pad(signal, (pad_left, pad_right), mode='constant')
        return [signal]
    
    # If the signal is longer than the desired length, split into 3-second windows
    elif len(signal) > max_length:
        # first cut the signal so it will split in multiples of 3
        signal = signal[:-(len(signal) % max_length)]
        segments = []
        for start in range(0, len(signal), max_length):
            end = start + max_length
            segment = signal[start:end]
            if len(segment) < max_length:
                segment = np.pad(segment, (0, max_length - len(segment)), mode='constant')
            if len(segment) >= target_sr:  # Ensure segment is at least 1 second long
                if len(segment) < max_length:
                    padding = max_length - len(segment)
                    pad_left = padding // 2
                    pad_right = padding - pad_left
                    segment = np.pad(segment, (pad_left, pad_right), mode='constant')
                segments.append(segment)

        return segments
    
    # If the signal is exactly the desired length, return as is
    else:
        return [signal]

def classify_audio(audio_clip):
    #If runs folder exists, delete it
    if os.path.exists("runs"):
        shutil.rmtree("runs")

    audio_signal = preprocess_audio(audio_clip)[0]

    # Ensure the audio signal has the correct shape
    audio_signal = np.array(audio_signal)
    audio_signal = audio_signal.reshape(1, -1)  # Reshape to (1, 144000) if needed

    # Perform object detection on the mel spectrogram image using the model
    y_predicted = model.predict(audio_signal)

    # Get the predicted classes and confidence
    y_pred_classes = np.argmax(y_predicted, axis=1)
    y_pred_confidence = np.max(y_predicted, axis=1)

    y_pred = np.vstack((y_pred_classes, y_pred_confidence))

    return y_pred

def process_prediction(prediction):
    # Get the class name and confidence and return as a string
    class_name = list(classes_mapping.keys())[list(classes_mapping.values()).index(prediction[0])]
    confidence = prediction[1]
    return f"{class_name}: {confidence.item():.2f}"

def predict(audio_clip, confidence_threshold=0.5):
    mel_spectrogram_image = update_mel_spectrogram(audio_clip)
    predictions_result = classify_audio(audio_clip)
    processed_prediction = process_prediction(predictions_result)
    return mel_spectrogram_image, processed_prediction

# Create Gradio interface with custom layout
audio_clip = gr.Audio(label="Upload Audio Clip", type="filepath")
mel_spectrogram = gr.Image(label="Mel Spectrogram")
confidence_threshold = gr.Slider(label="Confidence Threshold", minimum=0.0, maximum=1.0, value=0.5, step=0.05)
predictions = gr.Textbox(label="Predictions", value="Predictions will be placed here")

iface = gr.Interface(
    fn=predict,
    inputs=[audio_clip, confidence_threshold],
    outputs=[mel_spectrogram, predictions],
    title="Audio Classifier",
)

iface.launch()
