from ultralytics import YOLO
import torch

def train_yolo_model(data_yaml, model_path, epochs, img_size, batch_size, output_dir):
    # Load the model
    model = YOLO(model_path)  # Replace with the correct model path if necessary
    
    # Specify the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Print device information
    print(f"Using device: {device}")
    
    # Train the model
    model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        device=device,
        project=output_dir  # Specifies the output directory for the trained model
    )


if __name__ == '__main__':
    # Define parameters
    data_yaml = 'data.yaml'
    model_path = 'yolov8n.pt'
    epochs = 9
    img_size = 620  
    batch_size = 16
    output_dir = './models/yolo/trained'  # Specify your custom output directory

    # Call the training function
    train_yolo_model(data_yaml, model_path, epochs, img_size, batch_size, output_dir)