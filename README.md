# Malaria Parasite Detection using DEtection TRansformer (DETR)

## Project Overview
This project focuses on automating malaria diagnosis by identifying parasites in blood smear images. The solution is based on the **DETR (DEtection TRansformer)** architecture with a **ResNet-50** backbone. It treats object detection as a direct set prediction problem, eliminating the need for hand-crafted components like non-maximum suppression (NMS) or anchor generation.



## Tech Stack
- **Framework:** PyTorch
- **Architecture:** DETR (ResNet-50 backbone)
- **Data Format:** COCO
- **Libraries:** OpenCV, Matplotlib, NumPy, Pandas, Scipy

## Data Pipeline
The implementation includes the following key stages:
1. **Image Resizing:** All input images were resized to **640x640** pixels with corresponding bounding box coordinate scaling.
2. **Annotation Conversion:** Custom JSON annotations were converted into the industry-standard **COCO format**.
3. **Class Consolidation:** To address class imbalance and improve stability, 7 initial categories were mapped into two primary classes: **Infected** (parasitized cells) and **Uninfected** (healthy cells).
4. **Augmentations:** A combination of `RandomHorizontalFlip`, `RandomSelect` (Resize and Crop), and `RandomGaussianBlur` was applied during training.



## Training Results
The model was evaluated on a test set, showing high precision in identifying healthy cells and solid performance in parasite detection.

### Metrics (Threshold = 0.5):
| Category | Precision | Recall | Average IoU |
| :--- | :--- | :--- | :--- |
| **Infected** | 0.8486 | 0.6006 | 0.8579 |
| **Uninfected** | 0.8869 | 0.9247 | 0.8439 |



## Inference Visualization
Below is a comparison between the Ground Truth labels and the model's predictions on test images.



## Project Structure
- `main.py`: Entry point for training and evaluation.
- `models/`: Implementation of the DETR architecture components.
- `datasets/`: Scripts for data loading and COCO structure formatting.
- `util/`: Utility functions for logging and distributed computing.
- `requirements.txt`: List of required dependencies.

## Setup and Usage

### Installation
```bash
pip install -r requirements.txt
```

## Training
To start the training process:
python main.py --dataset_file coco --data_path ./path_to_data --output_dir ./output

## Inference
The visualization script is integrated into the run_inference_and_show_gt block for evaluating specific images.

Example usage within the script logic
run_inference_and_show_gt(model_path='checkpoint.pth', image_dir='test_images')
