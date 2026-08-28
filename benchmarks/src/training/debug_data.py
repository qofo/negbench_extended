import pandas as pd
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

from typing import List, Tuple, Callable
import torch
from training.video_utils.video_reader import DecordVideoReader, VideoReader
from training.video_utils.frame_sampler import FrameSampler, UniformFrameSampler

from training.video_utils.model import VideoCLIP

# Define the CsvCLassDataset class
class CsvCLassDataset(Dataset):
    def __init__(self, csv_file, transforms, sep=',', img_key='positive_filepath', target_key='target'):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            transforms (callable): Transform to be applied on a sample.
            sep (string): Separator used in the csv file.
            img_key (string): Column name for image file paths.
            target_key (string): Column name for target labels.
        """
        self.df = pd.read_csv(csv_file, sep=sep)
        self.transforms = transforms
        self.images = self.df[img_key].tolist()
        self.labels = self.df[target_key].tolist()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        images = Image.open(str(self.images[idx]))
        images = self.transforms(images)
        labels = self.labels[idx]
        return images, labels

class CsvVideoCaptionDataset(Dataset):
    def __init__(self, csv_file: str, transforms: Callable, max_frames: int = 4, video_key='filepath', caption_key='captions'):
        """
        Dataset for video captioning or retrieval tasks.

        Args:
            csv_file (str): Path to the CSV file with video paths and captions.
            transforms (callable): Transform to be applied on a sample of video frames.
            max_frames (int): Maximum number of frames to sample from each video. Defaults to 4.
            video_key (str): Column name for video file paths.
            caption_key (str): Column name for captions.
        """
        self.df = pd.read_csv(csv_file)
        self.transforms = transforms
        self.frame_sampler = UniformFrameSampler(max_frames=max_frames)
        self.video_key = video_key
        self.caption_key = caption_key

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        """
        Retrieves the video frames and corresponding captions for a given index.

        Args:
            idx (int): Index of the item to be retrieved.

        Returns:
            Tuple[torch.Tensor, List[str]]: A tuple containing:
                - transformed_frames (torch.Tensor): A tensor of shape (N, C, H, W) where:
                    - N is the number of sampled frames.
                    - C is the number of channels (typically 3 for RGB).
                    - H is the height of the frames.
                    - W is the width of the frames.
                The frames have been loaded from the video, permuted to match PyTorch's channel ordering, 
                and transformed using the specified transformations.
                
                - captions (List[str]): A list of strings, where each string is a caption corresponding to the video.
                The captions are extracted from the CSV file and are typically associated with the video in the dataset.
        """
        video_path = self.df.iloc[idx][self.video_key]

        # Captions follow the format of a list of strings (could be one string)
        captions = eval(self.df.iloc[idx][self.caption_key])

        # Load and sample frames from the video
        frames = self._load_and_sample_video(video_path)

        # Permute the frames to match PyTorch's (N, C, H, W) format
        frames = frames.permute(0, 3, 1, 2)  # Change from (N, H, W, C) to (N, C, H, W)

        # Apply transformations to the frames
        transformed_frames = self.transforms(frames)

        return transformed_frames, captions

    def _load_and_sample_video(self, video_path: str) -> torch.Tensor:
        # Initialize a VideoReader for the given video path
        video_reader = VideoReader.from_path(video_path)

        # Get the start and end frames for sampling
        start_frame = 0
        end_frame = len(video_reader) - 1

        # Sample the frames using the provided FrameSampler
        frame_indices = self.frame_sampler(start_frame, end_frame, video_reader.get_avg_fps())

        # Retrieve the frames from the video
        frames = video_reader(frame_indices)
        
        return frames

# if __name__ == '__main__':
#     # Set to True to test the video dataset, False to test the image dataset
#     debug_video_dataset = True

#     if debug_video_dataset:
#         data_transforms = transforms.Compose([transforms.Resize((224, 224), antialias=True)])
#         frame_sampler = UniformFrameSampler(max_frames=4) # Uniformly sample 4 frames from the video (following the FitCLIP paper)
#         csv_file = '/data/healthy-ml/scratch/kumail/projects/cs/negation/synthetic_dataset/negation_processing_scripts/data/video/MSRVTT/val_set.csv'
#         video_dataset = CsvVideoCaptionDataset(csv_file=csv_file, transforms=data_transforms, frame_sampler=frame_sampler)

#         # Test accessing the first item
#         first_item = video_dataset[0]

#         # Print the shape of the frames and the captions
#         print("First item frames shape:", first_item[0].shape)
#         print("First item captions:", first_item[1])
#     else:
#         # Define the path to the CSV file
#         dummy_csv_file = '/data/healthy-ml/scratch/kumail/projects/cs/negation/synthetic_dataset/mixtral_synthetic_dataset_zeroshot.csv'

#         # Define transformations
#         data_transforms = transforms.Compose([transforms.Resize((224, 224), antialias=True), transforms.ToTensor()])

#         # Instantiate the dataset
#         synthetic_dataset = CsvCLassDataset(csv_file=dummy_csv_file, transforms=data_transforms)

#         # Test accessing the first item
#         first_item = synthetic_dataset[0]
#         print("First item:", first_item)

from typing import Optional, Union, Sequence
from open_clip import create_model_and_transforms
import open_clip
class VideoLoader:
    def __init__(self, video_path: str, start_time: Optional[float] = None, end_time: Optional[float] = None):
        self.video_path = video_path
        self.video_reader = VideoReader.from_path(video_path)
        self.frame_sampler = UniformFrameSampler(max_frames=4)
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

# Example usage:
video_loader = VideoLoader("/data/healthy-ml/scratch/kumail/projects/cs/negation/synthetic_dataset/negation_processing_scripts/data/video/MSRVTT/videos/all/video9838.mp4")
video_tensor = video_loader.load_video_as_tensor()
print(video_tensor.shape)

model, _, preprocess = create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k', video=True)
tokenizer = open_clip.get_tokenizer('ViT-B-32')

image = preprocess(video_tensor)
print(image.shape)
# add a batch dimension
image = image.unsqueeze(0)
text = tokenizer(["a cartoon clip of pokemon dancing", "a woman talking to a man in a hood", "a 3d animation of a cabinet with plates"])

with torch.no_grad(), torch.cuda.amp.autocast():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

print("Label probs:", text_probs)  # prints: [[1., 0., 0.]]

csv_file = '/data/healthy-ml/scratch/kumail/projects/cs/negation/synthetic_dataset/negation_processing_scripts/data/video/MSRVTT/val_set.csv'
video_dataset = CsvVideoCaptionDataset(csv_file=csv_file, transforms=preprocess)

first_video, captions = video_dataset[0]
print("First video frames shape:", first_video.shape)
print("First video captions:", captions)
text = tokenizer(["a cartoon clip of pokemon dancing", "a woman talking to a man in a hood", "a 3d animation of a cabinet with plates"])

with torch.no_grad(), torch.cuda.amp.autocast():
    video_features = model.encode_image(first_video.unsqueeze(0))
    video_features /= video_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    video_probs = (100.0 * video_features @ text_features.T).softmax(dim=-1)

print("Label probs:", video_probs)  # prints: [[1., 0., 0.]]

# command to zip all files in a directory: zip -r archive_name.zip directory_to_compress