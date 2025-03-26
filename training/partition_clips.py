import argparse
import os
from pathlib import Path

import pandas as pd
import torchaudio
from tqdm import tqdm

min_length_sec = 8.1
max_segments_per_clip = 5

parser = argparse.ArgumentParser(description='Process audio clips.')
parser.add_argument('--data_dir',
                    type=Path,
                    help='Path to the directory containing audio files',
                    default='./training/example_audios')
parser.add_argument('--output_dir',
                    type=Path,
                    help='Path to the output tsv file',
                    default='./training/example_output/clips.tsv')
parser.add_argument('--start', type=int, help='Start index for processing files', default=0)
parser.add_argument('--end', type=int, help='End index for processing files', default=-1)
args = parser.parse_args()

data_dir = args.data_dir
output_dir = args.output_dir
start = args.start
end = args.end

output_data = []

blacklisted = 0
if end == -1:
    end = len(os.listdir(data_dir))
audio_files = sorted(os.listdir(data_dir))[start:end]
print(f'Processing {len(audio_files)} files from {start} to {end}')

for audio_file in tqdm(audio_files):
    audio_file_path = data_dir / audio_file
    audio_name = audio_file_path.stem

    waveform, sample_rate = torchaudio.load(audio_file_path)

    # waveform: (1/2) * length
    if waveform.shape[1] < sample_rate * min_length_sec:
        continue

    # try to partition the audio into segments, each with length of min_length_sec
    segment_length = int(sample_rate * min_length_sec)
    total_length = waveform.shape[1]
    num_segments = min(max_segments_per_clip, total_length // segment_length)
    if num_segments > 1:
        segment_interval = (total_length - segment_length) // (num_segments - 1)
    else:
        segment_interval = 0

    for i in range(num_segments):
        start_sample = i * segment_interval
        end_sample = start_sample + segment_length
        audio_id = f'{audio_name}_{i}'
        output_data.append((audio_id, audio_name, start_sample, end_sample))

output_dir.parent.mkdir(parents=True, exist_ok=True)
output_df = pd.DataFrame(output_data, columns=['id', 'name', 'start_sample', 'end_sample'])
output_df.to_csv(output_dir, index=False, sep='\t')
