import os
import random
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("FastJobs/Visual_Emotional_Analysis")

# View the structure (optional for checking the dataset)
print(dataset)

# Access a specific sample to view its structure (optional for checking one sample)
sample = dataset['train'][0]
print(sample)

# Define paths for saving images and labels
saved_images_train_path = 'yolo/datasets/trainings/fastjobs/images/train'
saved_images_val_path = 'yolo/datasets/trainings/fastjobs/images/val'
saved_labels_train_path = 'yolo/datasets/trainings/fastjobs/labels/train'
saved_labels_val_path = 'yolo/datasets/trainings/fastjobs/labels/val'

# Create directories if they do not exist
os.makedirs(saved_images_train_path, exist_ok=True)
os.makedirs(saved_images_val_path, exist_ok=True)
os.makedirs(saved_labels_train_path, exist_ok=True)
os.makedirs(saved_labels_val_path, exist_ok=True)

# Shuffle the dataset and split it into training and validation sets (80% train, 20% val)
data = list(dataset['train'])
random.shuffle(data)
split_index = int(len(data) * 0.8)  # 80% for training, 20% for validation
train_data = data[:split_index]
val_data = data[split_index:]

# Helper function to save images and labels
def save_image_and_label(dataset_split, image_path, label_path):
    for i, sample in enumerate(dataset_split):
        # Save the image
        image = sample['image']
        image_file_path = os.path.join(image_path, f'{i}.jpg')
        image.save(image_file_path)

        # Save the corresponding label in YOLO format with 5 columns
        label = sample['label']
        label_file_path = os.path.join(label_path, f'{i}.txt')
        
        # Bounding box covering the entire image
        x_center = 0.5
        y_center = 0.5
        width = 1.0
        height = 1.0

        with open(label_file_path, 'w') as label_file:
            # Write the label in the format: <class_id> <x_center> <y_center> <width> <height>
            label_file.write(f'{label} {x_center} {y_center} {width} {height}\n')

        # Print status to track progress
        print(f"Saved image: {image_file_path} and label: {label_file_path}")

# Save training data (80%)
save_image_and_label(train_data, saved_images_train_path, saved_labels_train_path)

# Save validation data (20%)
save_image_and_label(val_data, saved_images_val_path, saved_labels_val_path)

print("Dataset conversion and split completed!")
