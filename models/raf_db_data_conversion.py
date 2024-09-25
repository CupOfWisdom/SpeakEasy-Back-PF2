import pandas as pd
import os
from pathlib import Path
from PIL import Image

# Define paths
print(os.getcwd())

train_csv_dir = 'C:\\Users\\Administrator\\Documents\\University\\eigth-semester\\tcc\\speakeasy\\data\\training\\raf-db\\train_labels.csv'
test_csv_dir = 'C:\\Users\\Administrator\\Documents\\University\\eigth-semester\\tcc\\speakeasy\\data\\training\\raf-db\\test_labels.csv'

data_dir = '..\\data\\training\\raf-db\\DATASET'
train_dir = os.path.join(data_dir, 'train')
test_dir = os.path.join(data_dir, 'test')
output_dir = './yolo/trainings/raf-db'
Path(output_dir, 'images', 'train').mkdir(parents=True, exist_ok=True)
Path(output_dir, 'images', 'test').mkdir(parents=True, exist_ok=True)
Path(output_dir, 'labels', 'train').mkdir(parents=True, exist_ok=True)
Path(output_dir, 'labels', 'test').mkdir(parents=True, exist_ok=True)

def convert_folder_to_yolo_format(input_folder, output_images_dir, output_labels_dir):
    for class_id in os.listdir(input_folder):
        class_folder = os.path.join(input_folder, class_id)
        if os.path.isdir(class_folder):
            for image_file in os.listdir(class_folder):
                if image_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(class_folder, image_file)
                    img = Image.open(img_path)
                    img_width, img_height = img.size
                    
                    # Save image to output folder
                    new_image_path = os.path.join(output_images_dir, image_file)
                    img.save(new_image_path)
                    
                    # Create corresponding .txt file
                    txt_file = os.path.join(output_labels_dir, image_file.rsplit('.', 1)[0] + '.txt')
                    with open(txt_file, 'w') as f:
                        x_center = 0.5
                        y_center = 0.5
                        width = 1.0
                        height = 1.0
                        
                        # Write YOLO format line: <class_id> <x_center> <y_center> <width> <height>
                        f.write(f"{int(class_id)} {x_center} {y_center} {width} {height}\n")

def process_dataset(labels_csv, images_dir, output_images_dir, output_labels_dir):
    df = pd.read_csv(labels_csv)
    for _, row in df.iterrows():
        image_name = row['image']
        class_id = row['label']
        
        # Copy image to YOLO format directory
        img_path = os.path.join(images_dir, str(class_id), image_name)
        if os.path.exists(img_path):
            img = Image.open(img_path)
            img_width, img_height = img.size
            
            # Save image to output folder
            new_image_path = os.path.join(output_images_dir, image_name)
            img.save(new_image_path)
            
            # Create corresponding .txt file
            txt_file = os.path.join(output_labels_dir, image_name.rsplit('.', 1)[0] + '.txt')
            with open(txt_file, 'w') as f:
                x_center = 0.5
                y_center = 0.5
                width = 1.0
                height = 1.0
                
                # Write YOLO format line: <class_id> <x_center> <y_center> <width> <height>
                f.write(f"{int(class_id)} {x_center} {y_center} {width} {height}\n")

# Convert train and test datasets
convert_folder_to_yolo_format(train_dir, os.path.join(output_dir, 'images', 'train'), os.path.join(output_dir, 'labels', 'train'))
convert_folder_to_yolo_format(test_dir, os.path.join(output_dir, 'images', 'test'), os.path.join(output_dir, 'labels', 'test'))

# Process CSV files
process_dataset(train_csv_dir, train_dir, os.path.join(output_dir, 'images', 'train'), os.path.join(output_dir, 'labels', 'train'))
process_dataset(test_csv_dir, test_dir, os.path.join(output_dir, 'images', 'test'), os.path.join(output_dir, 'labels', 'test'))

print("Conversion completed!")
