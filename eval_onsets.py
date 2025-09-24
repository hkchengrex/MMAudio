# Modified from https://github.com/XYPB/CondFoleyGen/blob/main/predict_onset.py

import argparse
import copy
import os
from pathlib import Path

import librosa
import numpy as np
from sklearn.metrics import (average_precision_score, f1_score, precision_recall_curve)
from tqdm import tqdm

sample_rate = 22050
conf_interval = int(0.05 * 22050)
duration = 8


def onset_nms(onsets, wav_norm, window=0.05):
    confidence = [np.max(wav_norm[o - conf_interval:o + conf_interval]) for o in onsets]

    onset_remain = onsets.tolist()
    output = []
    sorted_idx = np.argsort(confidence)[::-1]
    for idx in sorted_idx:
        cur = onsets[idx]
        if cur not in onset_remain:
            continue
        output.append(cur)
        onset_remain.remove(cur)
        for o in onset_remain:
            if abs(cur - o) < window * sample_rate:
                onset_remain.remove(o)
    return np.array(sorted(output))


def predict_audio(audio_path: Path, delta: float) -> np.ndarray:
    wav, _ = librosa.load(audio_path, sr=sample_rate)
    wav = wav[:duration * sample_rate]
    onsets = librosa.onset.onset_detect(y=wav, sr=sample_rate, units='samples', delta=delta)
    wav_norm = (wav - wav.min()) / (wav.max() - wav.min() + 1e-6)

    return onsets, wav_norm


def read_gt(gt_file: Path) -> np.ndarray:
    all_times = []
    with open(gt_file, 'r') as f:
        lines = f.readlines()
        for l in lines:
            time = float(l.split(' ')[0])
            if time >= duration:
                break
            all_times.append(time)
    return np.array(all_times)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=Path)
    parser.add_argument('--gt_dir', type=Path)
    parser.add_argument('--delta', type=float, default=0.3)
    args = parser.parse_args()

    input_dir = args.input_dir
    gt_dir = args.gt_dir
    delta = args.delta

    overall_acc = 0
    overall_ap = 0
    overall_f1 = 0

    audio_files = sorted(os.listdir(input_dir))
    audio_files = [f for f in audio_files if f.endswith('.flac') or f.endswith('.wav')]
    for audio_file in tqdm(audio_files):
        base_name = Path(audio_file).stem
        gt_name = base_name.replace('_denoised', '_times')
        gt_file = gt_dir / f'{gt_name}.txt'
        gt_times = read_gt(gt_file) * sample_rate

        onsets, wav_norm = predict_audio(input_dir / audio_file, delta)
        onsets = onset_nms(onsets, wav_norm)

        onsets_onuse = copy.deepcopy(onsets.tolist())
        onsets_res = [0 for _ in onsets_onuse]

        y_gt = []
        y_pred = []
        hit_cnt = 0
        for gt_onset in gt_times:
            diff = [abs(pred_onset - gt_onset) for pred_onset in onsets_onuse]
            idx_in_window = [idx for idx in range(len(onsets_onuse)) if diff[idx] < delta * 22050]
            if len(idx_in_window) == 0:
                y_gt.append(1)
                y_pred.append(0)
            else:
                conf_in_window = [wav_norm[onsets[idx]] for idx in idx_in_window]
                max_conf_idx = np.argsort(conf_in_window)[-1]
                match_idx = idx_in_window[max_conf_idx]
                conf = np.max(wav_norm[onsets_onuse[match_idx] -
                                       conf_interval:onsets_onuse[match_idx] + conf_interval])
                hit_cnt += 1
                y_gt.append(1)
                y_pred.append(conf)
                # y_pred.append(1)
                for i in range(len(onsets)):
                    if onsets[i] == onsets_onuse[match_idx]:
                        onsets_res[i] = 1
                onsets_onuse.remove(onsets_onuse[match_idx])
                if len(onsets_onuse) == 0:
                    break

        for o in onsets_onuse:
            y_gt.append(0)
            y_pred.append(np.max(wav_norm[o - conf_interval:o + conf_interval]))
            # y_pred.append(1)

        acc = hit_cnt / len(gt_times) if len(gt_times) != 0 else 0
        ap = average_precision_score(y_gt, y_pred)
        f1 = f1_score(y_gt, [1 if p > 0 else 0 for p in y_pred])
        # print(y_gt, y_pred, ap, f1)

        overall_acc += acc
        overall_ap += ap
        overall_f1 += f1

    overall_acc /= len(audio_files)
    overall_ap /= len(audio_files)
    overall_f1 /= len(audio_files)
    print(f'Overall accuracy: {overall_acc:.4f}')
    print(f'Overall AP: {overall_ap:.4f}')
    print(f'Overall F1: {overall_f1:.4f}')

    # write to file
    with open(input_dir / 'eval_results.txt', 'w') as f:
        f.write(f'Overall accuracy: {overall_acc:.4f}\n')
        f.write(f'Overall AP: {overall_ap:.4f}\n')
        f.write(f'Overall F1: {overall_f1:.4f}\n')


if __name__ == '__main__':
    main()
