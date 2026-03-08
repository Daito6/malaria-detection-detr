# Malaria Parasite Detection using DEtection TRansformer (DETR)

## Project Overview
This project implements an automated malaria diagnosis system using the **DETR (DEtection TRansformer)** architecture with a **ResNet-50** backbone. The model treats object detection as a direct set prediction problem, providing a streamlined pipeline for identifying parasites in blood smear images.

## Data Exploration and Analysis
The dataset consists of high-resolution microscopic images of blood cells. Below is a sample of the raw data with initial annotations.

<img width="1024" height="761" alt="image" src="https://github.com/user-attachments/assets/803ab68d-41cf-4ded-bb2f-f7e600ea1d81" />

### Dataset Distribution Analysis
The dataset contains **1,208 training images** and **120 test images**. Analysis of the category distribution reveals a significant data imbalance:

<img width="1228" height="906" alt="image" src="https://github.com/user-attachments/assets/c532a11b-dd16-4406-9022-f475f7627004" />


The "red blood cell" class overwhelmingly dominates the dataset, accounting for over **93% of total annotations** in the training set. Minority classes like "gametocyte" or "leukocyte" are significantly underrepresented, representing less than 0.2% of total boxes.

## Data Pipeline and Preprocessing
To prepare the data for the Transformer-based model, a custom resizing and scaling pipeline was implemented.

### Image Resizing
Images were resized from their original resolutions (e.g., 1944x1383) to a uniform **640x640** pixels. Bounding box coordinates were automatically scaled to maintain spatial accuracy.

<img width="1024" height="630" alt="image" src="https://github.com/user-attachments/assets/a8083bc0-c1dd-403a-a3dc-838a9b21e699" />

### Class Consolidation and Remapping
Due to the significant imbalance and overwhelming dominance of certain categories in our dataset — especially the "red blood cell" class, which accounts for the vast majority of annotations — we decided to simplify our classification task. Instead of working with the original 7 fine-grained categories, we consolidated them into two broader classes: **"infected"** and **"uninfected"**.

This re-mapping allows us to:
* Reduce the class imbalance problem, which can negatively impact model performance.
* Focus the model's learning on the key distinction between healthy and infected cells, which is often the primary clinical objective.
* Improve generalization and robustness, especially when the minority classes are underrepresented.

## Training and Evaluation
The model was trained using the AdamW optimizer with a learning rate of $1e-4$ and a ResNet-50 backbone learning rate of $1e-5$.

### Training Metrics
Below are the training and validation curves for Total Loss, Classification Loss (CE), Bounding Box Loss, and GIoU Loss, along with Mean Average Precision (mAP) trends.

<img width="1554" height="866" alt="image" src="https://github.com/user-attachments/assets/a0d79700-6a0a-4da2-82fe-2b27bcc61a92" />
<img width="1568" height="896" alt="image" src="https://github.com/user-attachments/assets/3fabfdcb-b927-4c5c-8efc-db257be48b17" />
<img width="1550" height="858" alt="image" src="https://github.com/user-attachments/assets/802f76d2-9261-460a-a3c4-9b5417d63791" />
<img width="1542" height="850" alt="image" src="https://github.com/user-attachments/assets/556a503d-1b82-4cf2-9b11-b91fd564826b" />
<img width="1024" height="551" alt="image" src="https://github.com/user-attachments/assets/6095c46f-09a2-40f2-a704-1da6b4393d99" />


### Final Performance Metrics
The model achieved high accuracy in identifying uninfected cells while maintaining solid precision for infected instances:

| Category | Precision | Recall | Average IoU |
| :--- | :--- | :--- | :--- |
| **Infected** | 0.8486 | 0.6006 | 0.8579 |
| **Uninfected** | 0.8869 | 0.9247 | 0.8439 |

## Inference Results
The final model performance was validated by comparing Ground Truth annotations against model predictions on unseen test data.

<img width="1024" height="520" alt="image" src="https://github.com/user-attachments/assets/90222b84-87bc-4a36-8990-69f822ffe873" />
<img width="1024" height="533" alt="image" src="https://github.com/user-attachments/assets/06acd54c-b68c-4d52-aaee-2cbd335a709a" />
<img width="1575" height="799" alt="image" src="https://github.com/user-attachments/assets/bbbb77e6-dd88-46d8-a2b8-d14dd33b776a" />
<img width="1570" height="799" alt="image" src="https://github.com/user-attachments/assets/67fbbaa8-1666-49fd-853b-ca17bf939626" />


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
