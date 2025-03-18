import os
import time
import argparse
from typing import List, Tuple

import cv2
import torch
import matplotlib.pyplot as plt
from PIL import Image
import open_clip
from timm.utils import reparameterize_model
import numpy as np

# Check CUDA availability and set the device (GPU if available, otherwise CPU)
cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")
print(f"Torch device: {device}")

# Load MobileCLIP model and preprocessing transforms
model, _, preprocess = open_clip.create_model_and_transforms('MobileCLIP-S1', pretrained='datacompdr')
tokenizer = open_clip.get_tokenizer('MobileCLIP-S1')

# Set model to evaluation mode, reparameterize for efficiency, and move it to the selected device
model.eval()
model = reparameterize_model(model)
model.to(device)


def classify_image(img: np.ndarray, labels_list: List[str]) -> Tuple[str, float]:
    """
    Classify an image using MobileCLIP.

    This function preprocesses the input image, tokenizes the provided text prompts,
    extracts features from both image and text, computes the similarity, and returns the label with
    the highest probability along with the probability value.

    Args:
        img (numpy.ndarray): Input image in RGB format.
        labels_list (list): List of labels to classify against.

    Returns:
        tuple: A tuple containing the predicted label (str) and the probability (float).
    """
    # Convert the image from a NumPy array to a PIL image, preprocess it, add batch dimension, and move to device.
    preprocessed_img = preprocess(Image.fromarray(img)).unsqueeze(0).to(device)

    # Tokenize the labels inside the function and move tokens to the device.
    text = tokenizer(labels_list).to(device)

    # Disable gradient calculation and enable automatic mixed precision (if using GPU)
    with torch.no_grad(), torch.cuda.amp.autocast():

        # Extract features from the image using the model.
        image_features = model.encode_image(preprocessed_img)

        # Extract text features from the tokenized text.
        text_features = model.encode_text(text)

        # Normalize image and text features to unit vectors.
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        # Compute the similarity (dot product) and apply softmax to obtain probabilities.
        text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

    # Get the label with the highest probability from the provided label list.
    selected_label = labels_list[text_probs.argmax(dim=-1)]
    selected_prob = text_probs.max(dim=-1)[0].item()

    return selected_label, selected_prob


def plot_results(results: List[Tuple[np.ndarray, str, float, float]]) -> None:
    """
    Plot the classification results.

    This function creates a horizontal plot for each image in the results, displaying the image along with
    its predicted label, probability, and processing time.

    Args:
        results (list): List of tuples (img, label, probability, elapsed_time).
    """
    # Create subplots with one image per subplot.
    fig, axes = plt.subplots(1, len(results), figsize=(len(results) * 5, 5))

    # If there is only one image, make axes a list to handle it uniformly.
    if len(results) == 1:
        axes = [axes]

    # Iterate over results and plot each one.
    for ax, (img, label, prob, elapsed_time) in zip(axes, results):
        ax.imshow(img)
        ax.set_title(f"Label: {label},\nProbability: {prob:.2%},\nTime: {elapsed_time:.2f}s")
        ax.axis('off')

    plt.tight_layout()
    plt.show()


def main(data_folder: str, labels_list: List[str]) -> None:
    """
    Process images and perform zero-shot image classification.

    This function processes each image in the specified folder, classifies them using MobileCLIP,
    and then plots the results.

    Args:
        data_folder (str): Path to the folder containing input images.
        labels_list (List[str]): List of labels to classify against.
    """
    results: List[Tuple[np.ndarray, str, float, float]] = []

    for image_file in os.listdir(data_folder):
        image_path = os.path.join(data_folder, image_file)
        # Read the image using OpenCV.
        img = cv2.imread(image_path)
        # Skip files that are not valid images.
        if img is None:
            print(f"Warning: Unable to read image {image_file}. Skipping.")
            continue

        # Convert the image from BGR (OpenCV default) to RGB.
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        start_time = time.time()
        selected_label, selected_prob = classify_image(img, labels_list)
        elapsed_time = time.time() - start_time

        print(f"{image_file} - Label: {selected_label}, Prob: {selected_prob:.2%} (Time: {elapsed_time:.2f}s)")

        results.append((img, selected_label, selected_prob, elapsed_time))

    plot_results(results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Zero-Shot Image Classification with MobileCLIP"
    )
    parser.add_argument(
        '--data_folder',
        type=str,
        default="data",
        help="Path to folder containing input images"
    )
    parser.add_argument(
        '--labels_list',
        type=str,
        default="cat,dog,car",
        help="Comma-separated list of labels to classify against (e.g., 'cat,dog,car')"
    )
    args = parser.parse_args()

    data_folder: str = args.data_folder
    labels_list: List[str] = [label.strip() for label in args.labels_list.split(',') if label.strip()]

    main(data_folder, labels_list)