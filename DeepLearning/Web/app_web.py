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
    if isinstance(audio_clip, str):
        y, sr = librosa.load(audio_clip, sr=None)
    else:
        y = audio_clip
        sr = 48000  # Assuming the sample rate is 48000 Hz

    fmin = 1  # Minimum frequency (0 Hz)
    fmax = 48000  # Maximum frequency (32000 Hz)
    fig, ax = plt.subplots(figsize=(12, 6))  # Set the background color to black
    D = librosa.amplitude_to_db(librosa.stft(y), ref=np.max)
    librosa.display.specshow(D, sr=sr, x_axis="time", y_axis="log", fmin=fmin, fmax=fmax, ax=ax)  # Specify frequency range
    ax.axis('off')  # Remove axes
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, transparent=True)
    buf.seek(0)
    mel_spectrogram_image = Image.open(buf)
    return mel_spectrogram_image

def update_mel_spectrogram(audio_clip):
    mel_spectrogram_image = audio_to_mel_spectrogram(audio_clip)
    mel_spectrogram_image = mel_spectrogram_image.convert("RGB")
    return mel_spectrogram_image

def preprocess_audio(file_path, target_sr=48000, duration=3):
    signal, sr = librosa.load(file_path, sr=target_sr, mono=True)
    max_length = target_sr * duration  # e.g., 48000 * 3
    if len(signal) < max_length:
        padding = max_length - len(signal)
        pad_left = padding // 2
        pad_right = padding - pad_left
        signal = np.pad(signal, (pad_left, pad_right), mode='constant')
        return [signal]
    elif len(signal) > max_length:
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
    else:
        return [signal]

def classify_audio(audio_clip):
    if os.path.exists("runs"):
        shutil.rmtree("runs")
    audio_signal = preprocess_audio(audio_clip)[0]
    audio_signal = np.array(audio_signal)
    audio_signal = audio_signal.reshape(1, -1)  # Reshape to (1, 144000) if needed
    y_predicted = model.predict(audio_signal)
    y_pred_classes = np.argmax(y_predicted, axis=1)
    y_pred_confidence = np.max(y_predicted, axis=1)
    y_pred = np.vstack((y_pred_classes, y_pred_confidence))
    return y_pred

def process_prediction(prediction):
    class_name = list(classes_mapping.keys())[list(classes_mapping.values()).index(prediction[0])]
    confidence = prediction[1]
    return f"{class_name}: {confidence.item():.2f}"

current_segment_index = 0
segments = []

def predict(audio_clip, direction=None):
    global current_segment_index, segments
    if direction is None:
        segments = preprocess_audio(audio_clip)
        current_segment_index = 0
    elif direction == "left":
        current_segment_index = max(0, current_segment_index - 1)
    elif direction == "right":
        current_segment_index = min(len(segments) - 1, current_segment_index + 1)
    mel_spectrogram_image = audio_to_mel_spectrogram(segments[current_segment_index])
    predictions_result = classify_audio_segment(segments[current_segment_index])
    processed_prediction = process_prediction(predictions_result)
    return mel_spectrogram_image, processed_prediction

def classify_audio_segment(audio_segment):
    audio_signal = np.array(audio_segment)
    audio_signal = audio_signal.reshape(1, -1)  # Reshape to (1, 144000) if needed
    y_predicted = model.predict(audio_signal)
    y_pred_classes = np.argmax(y_predicted, axis=1)
    y_pred_confidence = np.max(y_predicted, axis=1)
    y_pred = np.vstack((y_pred_classes, y_pred_confidence))
    return y_pred

def update_interface(audio_clip):
    return predict(audio_clip)

def navigate_left(audio_clip):
    return predict(audio_clip, direction="left")

def navigate_right(audio_clip):
    return predict(audio_clip, direction="right")

def main():
    with gr.Blocks() as iface:
        with gr.Row():
            with gr.Column():
                audio_clip = gr.Audio(label="Upload Audio Clip", type="filepath")
                # confidence_threshold = gr.Slider(label="Confidence Threshold", minimum=0.0, maximum=1.0, value=0.5, step=0.05)
            with gr.Column():
                mel_spectrogram = gr.Image(label="Mel Spectrogram")
                predictions = gr.Textbox(label="Predictions", value="Predictions will be placed here")
            
                with gr.Row():
                    left_button = gr.Button("←")
                    right_button = gr.Button("→")
            
            audio_clip.change(fn=update_interface, inputs=[audio_clip], outputs=[mel_spectrogram, predictions])
            left_button.click(fn=navigate_left, inputs=[audio_clip], outputs=[mel_spectrogram, predictions])
            right_button.click(fn=navigate_right, inputs=[audio_clip], outputs=[mel_spectrogram, predictions])
            
    iface.launch(share=True)

if __name__ == "__main__":
    main()