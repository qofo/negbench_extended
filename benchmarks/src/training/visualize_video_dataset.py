import argparse
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
from typing import Optional
import torch
from training.video_utils.video_reader import VideoReader
from training.video_utils.frame_sampler import UniformFrameSampler

class VideoLoader:
    def __init__(self, video_path: str, max_frames: int = 4, start_time: Optional[float] = None, end_time: Optional[float] = None):
        self.video_path = video_path
        self.video_reader = VideoReader.from_path(video_path)
        self.frame_sampler = UniformFrameSampler(max_frames=max_frames)
        self.start_time = start_time
        self.end_time = end_time

    def load_video_as_tensor(self) -> torch.Tensor:
        start_frame_idx = 0 if self.start_time is None else self.video_reader.time_to_indices(self.start_time).item()
        end_frame_idx = len(self.video_reader) - 1 if self.end_time is None else self.video_reader.time_to_indices(self.end_time).item()

        print(f"Loading frames from {start_frame_idx} to {end_frame_idx}")

        # Sample the frames using the provided FrameSampler
        frame_indices = self.frame_sampler(start_frame_idx, end_frame_idx, self.video_reader.get_avg_fps())
        frames = self.video_reader(frame_indices)

        return frames.permute(0, 3, 1, 2)

def visualize_mcq_task(sampled_df, rephrased_df, max_frames=4):
    for idx, row in sampled_df.iterrows():
        video_path = row['image_path']
        captions = [row[f'caption_{i}'] for i in range(4)]
        correct_answer = row['correct_answer']
        correct_answer_template = row['correct_answer_template']

        # Retrieve the corresponding rephrased captions from the rephrased dataframe
        rephrased_row = rephrased_df.loc[rephrased_df['image_path'] == video_path]
        rephrased_captions = [rephrased_row[f'caption_{i}'].values[0] for i in range(4)]

        # Load video frames using VideoLoader
        video_loader = VideoLoader(video_path, max_frames=max_frames)
        frames = video_loader.load_video_as_tensor()

        # Plot the frames
        fig, axs = plt.subplots(1, max_frames, figsize=(15, 5))

        for i, frame in enumerate(frames):
            axs[i].imshow(frame.permute(1, 2, 0).numpy())  # Convert from (C, H, W) to (H, W, C) and to numpy
            axs[i].axis('off')

        # Add captions below the plot, showing original and rephrased versions
        plt.figtext(0.5, 0.01, "\n".join([f"{i}: {captions[i]} --> {rephrased_captions[i]}" for i in range(4)]), ha="center", fontsize=12)

        # Highlight the correct caption
        plt.figtext(0.5, 0.85, f"{correct_answer}: {rephrased_captions[correct_answer]}", ha="center", fontsize=14, color='red')

        # Set the title of the plot
        plt.suptitle(correct_answer_template, fontsize=16)

        # Save the plot to a file
        # plt.savefig(f"visualization/mcq_mixtral/video_{idx}.png") # TODO: Uncomment this line to save the plots
        plt.savefig(f"visualization/mcq_llama/video_{idx}.png")
        plt.close(fig)  # Close the plot to free up memory

def visualize_retrieval_task(sampled_df, rephrased_df, max_frames=4):
    for idx, row in sampled_df.iterrows():
        video_path = row['filepath']
        templated_caption = eval(row['captions'])[0]

        # Retrieve the corresponding rephrased caption from the rephrased dataframe
        rephrased_row = rephrased_df.loc[rephrased_df['filepath'] == video_path]
        rephrased_caption = eval(rephrased_row['captions'].values[0])[0]

        # Load video frames using VideoLoader
        video_loader = VideoLoader(video_path, max_frames=max_frames)
        frames = video_loader.load_video_as_tensor()

        # Plot the frames
        fig, axs = plt.subplots(1, max_frames, figsize=(15, 5))

        for i, frame in enumerate(frames):
            axs[i].imshow(frame.permute(1, 2, 0).numpy())  # Convert from (C, H, W) to (H, W, C) and to numpy
            axs[i].axis('off')

        # Add the single caption with its rephrased version below the plot
        plt.figtext(0.5, 0.01, f"Original: {templated_caption}\nRephrased: {rephrased_caption}", ha="center", fontsize=12)

        # Save the plot to a file
        # plt.savefig(f"visualization/retrieval_mixtral/video_{idx}.png") # TODO: Uncomment this line to save the plots
        plt.savefig(f"visualization/retrieval_llama/video_{idx}.png")
        plt.close(fig)  # Close the plot to free up memory

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize video captions for either MCQ or retrieval tasks.")
    parser.add_argument('--task', type=str, required=True, choices=['mcq', 'retrieval'], help='The task type: "mcq" or "retrieval".')
    parser.add_argument('--max_frames', type=int, default=4, help='Maximum number of frames to sample from the video.')
    args = parser.parse_args()

    base_path = "/data/healthy-ml/scratch/kumail/projects/cs/negation/synthetic_dataset/negation_processing_scripts/data/video/MSRVTT"
    if args.task == 'mcq':
        # csv_file_path = f"{base_path}/negation/msr_vtt_mcq_templated.csv" # TODO: Change this to the path of the templated CSV
        # rephrased_csv_file_path = f"{base_path}/negation/msr_vtt_mcq_rephrased.csv"
        csv_file_path = f"{base_path}/negation/msr_vtt_mcq_templated_llama.csv"
        rephrased_csv_file_path = f"{base_path}/negation/msr_vtt_mcq_rephrased_llama.csv"
    elif args.task == 'retrieval':
        # csv_file_path = f"{base_path}/negation/msr_vtt_retrieval_templated.csv"
        # rephrased_csv_file_path = f"{base_path}/negation/msr_vtt_retrieval_rephrased.csv" # TODO: Change this to the path of the rephrased CSV
        csv_file_path = f"{base_path}/negation/msr_vtt_retrieval_templated_llama.csv"
        rephrased_csv_file_path = f"{base_path}/negation/msr_vtt_retrieval_rephrased_llama.csv"

    df = pd.read_csv(csv_file_path)
    rephrased_df = pd.read_csv(rephrased_csv_file_path)

    # Sample 10 random rows from the CSV
    sampled_df = df.sample(n=10)

    # Execute the visualization based on the task
    if args.task == 'mcq':
        visualize_mcq_task(sampled_df, rephrased_df, max_frames=args.max_frames)
    elif args.task == 'retrieval':
        visualize_retrieval_task(sampled_df, rephrased_df, max_frames=args.max_frames)

# example usage
# python visualize_video_dataset.py --task mcq --max_frames 4
# python visualize_video_dataset.py --task retrieval --max_frames 4