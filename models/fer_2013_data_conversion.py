import os
from pathlib import Path
from PIL import Image

# Define paths for the dataset
data_dir = '../data/training/fer-2013'  # Replace with the actual directory path
train_dir = os.path.join(data_dir, 'train')
test_dir = os.path.join(data_dir, 'test')

# Define output directories for YOLO format
output_dir = './yolo/datasets/trainings/fer-2013'  # Adapt this as needed
Path(output_dir, 'images', 'train').mkdir(parents=True, exist_ok=True)
Path(output_dir, 'images', 'test').mkdir(parents=True, exist_ok=True)
Path(output_dir, 'labels', 'train').mkdir(parents=True, exist_ok=True)
Path(output_dir, 'labels', 'test').mkdir(parents=True, exist_ok=True)

# Map emotion names (folder names) to numerical class IDs
emotion_to_class_id = {
    'angry': 0,
    'disgusted': 1,
    'fearful': 2,
    'happy': 3,
    'neutral': 4,
    'sad': 5,
    'surprised': 6
}

# Function to convert the images in the dataset to YOLO format
def convert_folder_to_yolo_format(input_folder, output_images_dir, output_labels_dir):
    for emotion, class_id in emotion_to_class_id.items():
        emotion_folder = os.path.join(input_folder, emotion)
        if os.path.isdir(emotion_folder):
            for image_file in os.listdir(emotion_folder):
                if image_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(emotion_folder, image_file)
                    img = Image.open(img_path)
                    img_width, img_height = img.size
                    
                    # Save image to output folder
                    new_image_path = os.path.join(output_images_dir, image_file)
                    img.save(new_image_path)
                    
                    # Create corresponding .txt file in YOLO format
                    txt_file = os.path.join(output_labels_dir, image_file.rsplit('.', 1)[0] + '.txt')
                    with open(txt_file, 'w') as f:
                        x_center = 0.5
                        y_center = 0.5
                        width = 1.0
                        height = 1.0
                        
                        # Write YOLO format line: <class_id> <x_center> <y_center> <width> <height>
                        f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

# Convert train and test datasets to YOLO format
convert_folder_to_yolo_format(train_dir, os.path.join(output_dir, 'images', 'train'), os.path.join(output_dir, 'labels', 'train'))
convert_folder_to_yolo_format(test_dir, os.path.join(output_dir, 'images', 'test'), os.path.join(output_dir, 'labels', 'test'))

print("Conversion to YOLO format completed!")
