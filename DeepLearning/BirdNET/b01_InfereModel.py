import os
import numpy as np
import pandas as pd
import librosa
import argparse
import time
import tensorflow as tf

# sudo mount -t drvfs E: /mnt/e

# python DeepLearning/BirdNET/b01_InfereModel.py --audio_path ../../../mnt/e/Seanimalmove/WOPAM\ DAY\ 2024/ --output_folder ../../../mnt/e/Seanimalmove/WOPAM_Inference/

# python DeepLearning/BirdNET/b01_InfereModel.py --audio_path ../../../mnt/e/SEANIMALMOVE/SYLENCE\ TARIFA\ 3_1min/  --output_folder ../../../mnt/e/SEANIMALMOVE/Predictions/

# Define class mapping
CLASSES_MAPPING = {
    'Background': 0,
    'BottlenoseDolphin': 1,
    'CommonDolphin': 2,
    'Fin_FinbackWhale': 3,
    "Grampus_Risso'sDolphin": 4,
    'HarborPorpoise': 5,
    'HumpbackWhale': 6,
    'KillerWhale': 7,
    'Long_FinnedPilotWhale': 8,
    'MinkeWhale': 9,
    'SpermWhale': 10,
    'StripedDolphin': 11
}

CLASSES_MAPPING = {
    'Background': 0,
    'BottlenoseDolphin': 1,
    'CommonDolphin': 2,
    "Grampus_Risso'sDolphin": 3,
    'HarborPorpoise': 4,
    'KillerWhale': 5,
    'Long_FinnedPilotWhale': 6,
    'StripedDolphin': 7
}

INV_CLASSES_MAPPING = {v: k for k, v in CLASSES_MAPPING.items()}

class AudioDataset:
    def __init__(self, audio_path, sample_rate=48000):
        self.audio_path = audio_path
        self.sample_rate = sample_rate
        self.files = []
        self._process_audio()

    def _process_audio(self):
        if os.path.isfile(self.audio_path):
            self.files = [self.audio_path]
        elif os.path.isdir(self.audio_path):
            self.files = [os.path.join(self.audio_path, f) for f in os.listdir(self.audio_path) if f.endswith('.wav')]
            print(f"Number of audio files to be processed: {len(self.files)}")
        else:
            raise ValueError("Invalid path. Provide a valid file or folder containing audio files.")        

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]
        return file

def load_model_keras():
    # model_path = "DeepLearning/BirdNET/Models/18_MarineMammals_NewBGManagement" 
    # model_path = "DeepLearning/BirdNET/Models/20_MarineMammalsDelphinidae"
    # model_path = "DeepLearning/BirdNET/Models/22_MarineMammalsDelphinidae_BetterCutsCIRCE"
    model_path = "DeepLearning/BirdNET/Models/23_MarineMammalsDelphinidae_WOPAMBG"
    # model_path = "DeepLearning/BirdNET/Models/14_MarineMammals"
    model = tf.keras.models.load_model(model_path)
    return model

def process_audio_segments(file, sample_rate=48000, segment_length=3):
    segments = []
    signal, sr = librosa.load(file, sr=sample_rate, mono=True)
    max_length = sample_rate * segment_length  # e.g., 48000 * 3
    # print(f"Signal shape: {signal.shape}, Sample rate: {sr}, signal length: {len(signal)}, max length: {target_sr * duration}")
    if len(signal) <= max_length:
        # print(f"Case 0")
        padding = max_length - len(signal)
        pad_left = padding // 2
        pad_right = padding - pad_left
        signal = np.pad(signal, (pad_left, pad_right), mode='constant')
        # print(f"Case 0: Signal shape: {signal.shape}, Sample rate: {sr}, signal length: {len(signal)}")
        segments.append((file, signal, 0, 3))
    elif len(signal) > max_length:
        # print(f"Case 1")
        # signal = signal[:-(len(signal) % max_length)]
        for start in range(0, len(signal), max_length):
            end = start + max_length
            segment = signal[start:end]
            if len(segment) < max_length:
                segment = np.pad(segment, (0, max_length - len(segment)), mode='constant')
            if len(segment) >= sample_rate:  # Ensure segment is at least 1 second long
                if len(segment) < max_length:
                    padding = max_length - len(segment)
                    pad_left = padding // 2
                    pad_right = padding - pad_left
                    segment = np.pad(segment, (pad_left, pad_right), mode='constant')
                # print(f"Case 1: Signal shape: {segment.shape}, Sample rate: {sr}, signal length: {len(segment)}")
                segments.append((file, segment, start // sample_rate, end // sample_rate))
    else:
        segments.append((file, signal, 0, 3))
    
    print(f"Number of segments: {len(segments)} in {len(signal) / sample_rate} seconds")
    return segments
    

def classify_audio(model, audio_clip):
    audio_clip = np.expand_dims(audio_clip, axis=0)
    probabilities = model.predict(audio_clip, verbose=0)[0]
    top_class = np.argmax(probabilities)
    return top_class, probabilities

def process_audio_files(audio_path, output_folder):
    dataset = AudioDataset(audio_path)
    model = load_model_keras()
    
    all_results = []
    total_preprocess_time = 0
    total_inference_time = 0
    total_audio_duration = 0

    # Check already existing files in output folder
    existing_files = [f for f in os.listdir(output_folder) if f.endswith('.csv')]

    # Create df for each file already processed
    df_combined = pd.DataFrame()
    for f in existing_files:
        df = pd.read_csv(os.path.join(output_folder, f))
        df_combined = pd.concat([df_combined, df])
        print(f"Loaded {f} already processed files with {len(df)} rows")

    for i, file in enumerate(dataset):
        if os.path.basename(file).replace(".wav", ".csv") not in existing_files:
            print(f"Processing {file} ({i + 1}/{len(dataset)})")

            start_preprocess = time.time()
            segments = process_audio_segments(file)
            end_preprocess = time.time()
            preprocess_time = end_preprocess - start_preprocess
            total_preprocess_time += preprocess_time

            start_inference = time.time()
            for original_file, segment, start_s, end_s in segments:
                top_class, probabilities = classify_audio(model, segment)

                result = {
                    "Path": original_file,
                    "Filename": os.path.basename(file),
                    "StartSecond": start_s,
                    "EndSecond": end_s,
                    "MainClassification": INV_CLASSES_MAPPING[top_class],
                    "ConfidenceScore": probabilities[top_class],
                    "ConfidenceVector": probabilities.tolist()
                }

                all_results.append(result)
            end_inference = time.time()
            inference_time = end_inference - start_inference
            total_inference_time += inference_time

            total_audio_duration += len(segments) * 3  # Assuming each segment is 3 seconds long

            print(f"Time taken to preprocess {file}: {preprocess_time:.2f} seconds")
            print(f"Time taken to infer {file}: {inference_time:.2f} seconds")

            output_csv = os.path.join(output_folder, os.path.splitext(os.path.basename(file))[0] + ".csv")
            df = pd.DataFrame([res for res in all_results if res["Filename"] == os.path.basename(file)])
            df.to_csv(output_csv, index=False)

    # Convert all_results to DataFrame before concatenating
    df_all_results = pd.DataFrame(all_results)
    df_combined = pd.concat([df_combined, df_all_results])

    combined_csv = os.path.join(output_folder, "predictions.csv")
    df_combined.to_csv(combined_csv, index=False)
    
    print(f"Combined results saved to {combined_csv}")

    total_files = len(dataset)
    total_time = total_preprocess_time + total_inference_time
    avg_preprocess_time = total_preprocess_time / total_files
    avg_inference_time = total_inference_time / total_files
    avg_total_time = total_time / total_files

    print(f"Total number of audio files: {total_files}")
    print(f"Total duration of audio: {total_audio_duration / 60:.2f} minutes")
    print(f"Total time for preprocessing: {total_preprocess_time:.2f} seconds")
    print(f"Total time for inference: {total_inference_time:.2f} seconds")
    print(f"Total time for preprocessing + inference: {total_time:.2f} seconds")
    print(f"Average time per file for preprocessing: {avg_preprocess_time:.2f} seconds")
    print(f"Average time per file for inference: {avg_inference_time:.2f} seconds")
    print(f"Average time per file for preprocessing + inference: {avg_total_time:.2f} seconds")

import os
import pandas as pd

def combine_csv_files(input_folder, output_file):
    # List all CSV files in the input folder
    csv_files = [f for f in os.listdir(input_folder) if f.endswith('.csv')]

    # Initialize an empty DataFrame to hold the combined data
    combined_df = pd.DataFrame()

    # Loop through each CSV file and append its content to the combined DataFrame
    for csv_file in csv_files:
        file_path = os.path.join(input_folder, csv_file)
        df = pd.read_csv(file_path)
        combined_df = pd.concat([combined_df, df], ignore_index=True)

    # Save the combined DataFrame to the output file
    combined_df.to_csv(output_file, index=False)
    print(f"Combined CSV saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_path", type=str, required=True, help="Path to audio file or folder containing audio files.")
    parser.add_argument("--output_folder", type=str, required=True, help="Path to save the output CSV files.")
    args = parser.parse_args()

    os.makedirs(args.output_folder, exist_ok=True)
    process_audio_files(args.audio_path, args.output_folder)


# if __name__ == "__main__":
#     input_folder = "../../../mnt/f/WOPAM DAY DL/"  # Replace with your input folder path
#     output_file = "../../../mnt/f/WOPAM DAY DL/combined.csv"  # Replace with your output file path
#     combine_csv_files(input_folder, output_file)