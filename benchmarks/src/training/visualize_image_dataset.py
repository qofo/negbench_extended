import argparse
import matplotlib
matplotlib.use("Agg")  # these run on headless GPU nodes
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import os
import textwrap

def visualize_text_image_data(df, base_path, num_samples=10):
    """
    Visualizes a random selection of images with their corresponding captions and negative captions.

    Args:
        df (pd.DataFrame): DataFrame containing the dataset with columns 'filepath', 'caption', and 'negative_captions'.
        base_path (str): Base path where the image files are located.
        num_samples (int): Number of random samples to visualize.
    """
    # Sample random rows from the DataFrame
    sampled_df = df.sample(n=num_samples)

    for idx, row in sampled_df.iterrows():
        image_path = os.path.join(base_path, row['filepath'])
        caption = row['caption']
        negative_captions = eval(row['negative_captions'])  # Convert string representation of list to a list

        # Load the image
        image = Image.open(image_path)

        # Plot the image
        plt.figure(figsize=(10, 10))  # Adjust the figure size
        plt.imshow(image)
        plt.axis('off')

        # Word wrapping for captions
        wrapped_caption = textwrap.fill(f"Original Caption: {caption}", width=80)
        wrapped_negative_captions = [textwrap.fill(f"Negative {i + 1}: {neg_caption}", width=80)
                                     for i, neg_caption in enumerate(negative_captions)]

        # Clear the plot and only show the image
        plt.clf()
        plt.imshow(image)
        plt.axis('off')

        # Add captions below the image
        captions_text = wrapped_caption + '\n' + '\n'.join(wrapped_negative_captions)
        plt.figtext(0.5, -0.2, captions_text, ha="center", fontsize=12, wrap=True)  # Position below the image

        # Save the plot to a file
        output_dir = "visualization/text_image/"
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(f"{output_dir}image_{idx}.png", bbox_inches='tight', pad_inches=0.5)
        plt.close()  # Close the plot to free up memory

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize image captions and their negative counterparts.")
    parser.add_argument('--num_samples', type=int, default=10, help='Number of samples to visualize.')
    args = parser.parse_args()

    base_path = "/data/healthy-ml/gobi1/data/cc3m/negation_dataset/filtered_objects/"
    csv_file_path = f"{base_path}/train_images_mixtral_neg_captions_0_33000.csv"

    # Load the dataset
    df = pd.read_csv(csv_file_path)

    # Visualize the sampled data
    visualize_text_image_data(df, base_path, num_samples=args.num_samples)

# Example usage:
# python visualize_text_image_dataset.py --num_samples 10
