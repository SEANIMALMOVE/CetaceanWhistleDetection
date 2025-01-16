import gradio as gr
import librosa

import pandas as pd
import numpy as np
from PIL import Image
import shutil
import os
from io import BytesIO
import matplotlib.pyplot as plt
import tensorflow as tf # type: ignore
from marine_mammals_classes_mapping import CLASSES_MAPPING_REDUCED_REAL as classes_mapping

# Define the available experiments
EXPERIMENTS = [
    # "6_WatkinsReducedClasses_07Train_015Val_015Test_Background_DataAugmentation",
    # "10_MarineMammals",
    # "11_MarineMammals_CIRCEBG",
    # "12_MarineMammals_AllDataset_ReduceDimNN"
    "16_MarineMammals_NewDSGenerator",
    "15_MarineMammals_NewDS",
    "14_MarineMammals"
]

# Load the initial model
EXPERIMENT = EXPERIMENTS[0]
model = tf.keras.models.load_model(f"DeepLearning/BirdNET/Models/{EXPERIMENT}/")


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
    librosa.display.specshow(D, sr=sr, x_axis="time", y_axis="mel", fmin=fmin, fmax=fmax, ax=ax)  # Specify frequency range
    ax.axis('off')  # Remove axes
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, transparent=True)
    buf.seek(0)
    mel_spectrogram_image = Image.open(buf)
    plt.close(fig)

    # Convert the mel spectrogram image to max 500 Hz
    fig, ax = plt.subplots(figsize=(12, 6))  # Set the background color to black
    D = librosa.amplitude_to_db(librosa.stft(y), ref=np.max)
    librosa.display.specshow(D, sr=sr, x_axis="time", y_axis="mel", fmin=0, fmax=500, ax=ax)  # Specify frequency range
    ax.axis('off')  # Remove axes
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, transparent=True)
    buf.seek(0)
    mel_spectrogram_image_500hz = Image.open(buf)
    plt.close(fig)

    return mel_spectrogram_image, mel_spectrogram_image_500hz

def update_mel_spectrogram(audio_clip):
    mel_spectrogram_image, mel_spectrogram_image_500hz = audio_to_mel_spectrogram(audio_clip)
    mel_spectrogram_image = mel_spectrogram_image.convert("RGB")
    mel_spectrogram_image_500hz = mel_spectrogram_image_500hz.convert("RGB")
    return mel_spectrogram_image, mel_spectrogram_image_500hz

def preprocess_audio(file_path, target_sr=48000, duration=3):
    signal, sr = librosa.load(file_path, sr=target_sr, mono=True)
    max_length = target_sr * duration  # e.g., 48000 * 3
    # print(f"Signal shape: {signal.shape}, Sample rate: {sr}, signal length: {len(signal)}, max length: {target_sr * duration}")
    if len(signal) <= max_length:
        # print(f"Case 0")
        padding = max_length - len(signal)
        pad_left = padding // 2
        pad_right = padding - pad_left
        signal = np.pad(signal, (pad_left, pad_right), mode='constant')
        # print(f"Case 0: Signal shape: {signal.shape}, Sample rate: {sr}, signal length: {len(signal)}")
        return [signal]
    elif len(signal) > max_length:
        # print(f"Case 1")
        # signal = signal[:-(len(signal) % max_length)]
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
                # print(f"Case 1: Signal shape: {segment.shape}, Sample rate: {sr}, signal length: {len(segment)}")
                segments.append(segment)
        return segments
    else:
        return [signal]
    
def predict(audio_clip, confidence_threshold=0.5, direction=None):
    global current_segment_index, segments
    if direction is None:
        segments = preprocess_audio(audio_clip)
        current_segment_index = 0
    elif direction == "left":
        current_segment_index = max(0, current_segment_index - 1)
    elif direction == "right":
        current_segment_index = min(len(segments) - 1, current_segment_index + 1)

    if current_segment_index < 0 or current_segment_index >= len(segments):
        raise IndexError("current_segment_index is out of range : ", current_segment_index, " for ", len(segments))
    
    mel_spectrogram_image, mel_spectrogram_image_500hz = audio_to_mel_spectrogram(segments[current_segment_index])
    predictions_result = classify_audio_segment(segments[current_segment_index])
    processed_prediction = process_prediction(predictions_result, confidence_threshold)
    current_segment_audio = (48000, segments[current_segment_index])  # Return a tuple with sample rate and audio data
    start_second = current_segment_index * 3
    end_second = start_second + 3
    segment_info = f"From second {start_second} to second {end_second}"
    print(segment_info, ":", processed_prediction)
    return mel_spectrogram_image, mel_spectrogram_image_500hz, processed_prediction, current_segment_audio, segment_info

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

def process_prediction(prediction, confidence_threshold):
    class_names = [list(classes_mapping.keys())[list(classes_mapping.values()).index(cls)] for cls in prediction[0]]
    confidences = prediction[1]
    
    if confidences[0] >= confidence_threshold:
        return f"{class_names[0]}: {confidences[0].item():.2f}"
    else:
        cumulative_confidence = 0
        other_predictions = []
        for i in range(len(class_names)):
            cumulative_confidence += confidences[i]
            other_predictions.append(f"{class_names[i]}: {confidences[i].item():.2f}")
            if cumulative_confidence >= confidence_threshold:
                break
        return "\n".join(other_predictions)

current_segment_index = 0
segments = []

def classify_audio_segment(audio_segment):
    audio_signal = np.array(audio_segment)
    audio_signal = audio_signal.reshape(1, -1)  # Reshape to (1, 144000) if needed
    y_predicted = model.predict(audio_signal)
    y_pred_classes = np.argsort(y_predicted, axis=1)[:, ::-1]
    y_pred_confidence = np.sort(y_predicted, axis=1)[:, ::-1]
    y_pred = np.vstack((y_pred_classes, y_pred_confidence))
    return y_pred

def generate_csv(audio_clip):
    global segments
    segments = preprocess_audio(audio_clip)
    rows = []
    for i, segment in enumerate(segments):
        y_predicted = model.predict(np.array([segment]))
        y_pred_classes = np.argmax(y_predicted, axis=1)
        y_pred_confidence = np.max(y_predicted, axis=1)
        predictions_vector = y_predicted.flatten().tolist()
        start_second = i * 3
        end_second = start_second + 3
        row = {
            'Filename': audio_clip,
            'StartSecond': start_second,
            'EndSecond': end_second,
            'ClassPrediction': list(classes_mapping.keys())[list(classes_mapping.values()).index(y_pred_classes[0])],
            'ConfidenceScore': y_pred_confidence[0],
            'PredictionsVector': predictions_vector
        }
        rows.append(row)
    df = pd.DataFrame(rows)
    csv_path = 'predictions.csv'
    df.to_csv(csv_path, index=False)
    return csv_path

def update_interface(audio_clip, confidence_threshold):
    return predict(audio_clip, confidence_threshold)

def navigate_left(audio_clip, confidence_threshold):
    return predict(audio_clip, direction="left", confidence_threshold=confidence_threshold)

def navigate_right(audio_clip, confidence_threshold):
    return predict(audio_clip, direction="right", confidence_threshold=confidence_threshold)

def update_experiment(experiment):
    global model
    model = tf.keras.models.load_model(f"DeepLearning/BirdNET/Models/{experiment}/")
    return f"Model {experiment} loaded."

def main():
    with gr.Blocks() as iface:
        with gr.Row():
            with gr.Column():
                model_selector = gr.Radio(label="Select Model", choices=EXPERIMENTS, value=EXPERIMENTS[0])
                audio_clip = gr.Audio(label="Upload Audio Clip", type="filepath")
                confidence_threshold = gr.Slider(minimum=0, maximum=1, value=0.5, label="Confidence Threshold")
                download_button = gr.Button("Download Predictions CSV")

            with gr.Column():
                segment_info = gr.Textbox(label="Segment Info", value="Segment info will be placed here")
                mel_spectrogram = gr.Image(label="Mel Spectrogram")
                mel_spectrogram_500hz = gr.Image(label="Mel Spectrogram (500 Hz)")
                predictions = gr.Textbox(label="Predictions", value="Predictions will be placed here")
                segment_audio = gr.Audio(label="Current Segment Audio", type="numpy")
            
                with gr.Row():
                    left_button = gr.Button("←")
                    right_button = gr.Button("→")

            with gr.Column():
                model_status = gr.Textbox(label="Model Status")
                download_file = gr.File()
            
            model_selector.change(fn=update_experiment, inputs=[model_selector], outputs=model_status)
            confidence_threshold.change(fn=update_interface, inputs=[audio_clip, confidence_threshold], outputs=[mel_spectrogram, mel_spectrogram_500hz, predictions, segment_audio, segment_info])
            audio_clip.change(fn=update_interface, inputs=[audio_clip, confidence_threshold], outputs=[mel_spectrogram, mel_spectrogram_500hz, predictions, segment_audio, segment_info])
            left_button.click(fn=navigate_left, inputs=[audio_clip, confidence_threshold], outputs=[mel_spectrogram, mel_spectrogram_500hz, predictions, segment_audio, segment_info])
            right_button.click(fn=navigate_right, inputs=[audio_clip, confidence_threshold], outputs=[mel_spectrogram, mel_spectrogram_500hz, predictions, segment_audio, segment_info])
            download_button.click(fn=generate_csv, inputs=[audio_clip], outputs=download_file)

    iface.launch(share=True, server_port=7863)

if __name__ == "__main__":
    main()