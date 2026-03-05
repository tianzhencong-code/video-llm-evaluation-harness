import os
import json
import torch
import decord
import numpy as np
from torch.utils.data import Dataset
from datasets import load_dataset


def timestamp_to_seconds(timestamp):
 # Split the timestamp into hours, minutes, and seconds
 parts = timestamp.split(':')
 
 # Convert hours, minutes, and seconds to integers
 hours = int(parts[0])
 minutes = int(parts[1])
 seconds = float(parts[2])
 
 # Calculate total seconds
 total_seconds = hours * 3600 + minutes * 60 + seconds
 
 return total_seconds


class LongVideoBenchDataset(Dataset):
 def __init__(self, data_path, data_name, max_num_frames=None, split='test', 
 force_re_download=False, trim_ratio=None):
 self.data_path = data_path
 self.data_name = data_name
 self.max_num_frames = max_num_frames
 self.split = split
 self.force_re_download = force_re_download
 self.trim_ratio = trim_ratio
 
 self.data = self._load_data()
 
 def _load_data(self):
 data = load_dataset(self.data_name, split=self.split)
 return data

    def get_w_subtitle(self, idx):
        sample = self.data[idx]
        video_path = os.path.join(self.data_path, sample['video_path'])
        subtitle_path = os.path.join(self.data_path, sample['subtitle_path'])
        question = sample['question']
        candidates = [sample['option0'], sample['option1'], sample['option2'], sample['option3']]
        answer = sample['correct_choice']
        
        if os.path.exists(subtitle_path):
            with open(subtitle_path, 'r') as f:
                subtitles = json.load(f)
        else:
            subtitles = None
        
        frames, frame_timestamps = self.load_video(video_path, sample.get('duration', None))
        
        return {
            'video_path': video_path,
            'frames': frames,
            'frame_timestamps': frame_timestamps,
            'subtitles': subtitles,
            'question': question,
            'candidates': candidates,
            'answer': answer,
            'duration': sample.get('duration', None),
            'domain': sample.get('domain', None),
        }

    def get_wo_subtitle(self, idx):
        sample = self.data[idx]
        video_path = os.path.join(self.data_path, sample['video_path'])
        question = sample['question']
        candidates = [sample['option0'], sample['option1'], sample['option2'], sample['option3']]
        answer = sample['correct_choice']
        
        frames, frame_timestamps = self.load_video(video_path, sample.get('duration', None))
        
        return {
            'video_path': video_path,
            'frames': frames,
            'frame_timestamps': frame_timestamps,
            'question': question,
            'candidates': candidates,
            'answer': answer,
            'duration': sample.get('duration', None),
            'domain': sample.get('domain', None),
        }

    def load_video(self, video_path, duration=None):
        vr = decord.VideoReader(video_path)
        fps = vr.get_avg_fps()
        total_frames = len(vr)
        
        if duration is None:
            duration = total_frames / fps
        
        if self.max_num_frames is not None and total_frames > self.max_num_frames:
            frame_indices = np.linspace(0, total_frames - 1, self.max_num_frames, dtype=int)
        else:
            frame_indices = np.arange(total_frames)
        
        if self.trim_ratio is not None:
            start = int(len(frame_indices) * self.trim_ratio)
            end = int(len(frame_indices) * (1 - self.trim_ratio))
            frame_indices = frame_indices[start:end]
        
        frames = vr.get_batch(frame_indices).asnumpy()
        frame_timestamps = [idx / fps for idx in frame_indices]
        
        return frames, frame_timestamps

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.get_wo_subtitle(idx)
