# Image Classification with CNNs

## **Project Overview**
This project involves developing and deploying an image classification system using **Convolutional Neural Networks (CNNs)** based on the **VGG16 architecture**. The model is designed to classify images into four categories: **dog, cat, man, and woman**. The system leverages **transfer learning** with pre-trained ImageNet weights and incorporates a robust end-to-end machine learning pipeline, from data ingestion to cloud deployment.

---

## **GitHub Repository**
Find the complete source code and resources for this project at the following link:  
[ImageClassification Repository](https://github.com/AbdulRasheed1011/ImageClassification)

---

## **Key Features**
- **Data Ingestion**: Automated workflows to validate and preprocess datasets, including removing corrupted images and handling inconsistencies.
- **Model Preparation**: Transfer learning with **VGG16** pre-trained on **ImageNet** while fine-tuning layers for the target classification task.
- **Training and Evaluation**:
  - Achieved **88.42% accuracy** and **0.32 loss** over **10 epochs**.
  - Configured hyperparameters such as:
    - **Image Size**: 224x224x3
    - **Batch Size**: 16
    - **Learning Rate**: 0.001
- **Data Augmentation**: Used rotation, flipping, shifting, and zooming techniques to enhance training generalization.
- **Cloud Deployment**: Deployed on **AWS EC2** using a **CI/CD pipeline** with **GitHub Actions**, serving the model as a **Flask API** for real-time inference.
- **Model Classes**: Classifies images into four categories: **cat, dog, man, and woman**.

---

## **Technology Stack**
- **Frameworks and Libraries**:
  - TensorFlow
  - Keras
  - PIL (Python Imaging Library)
  - Flask
- **Tools**:
  - GitHub Actions (CI/CD)
  - AWS EC2 for deployment
  - JSON for score saving
- **Programming Language**:
  - Python

---

## **Applications**
1. **E-Commerce**: Product categorization and personalized recommendations.
2. **Healthcare**: Identifying anomalies in medical imaging.
3. **Security**: Facial recognition and surveillance analysis.
4. **Social Media**: Content moderation and automatic tagging.
5. **Autonomous Vehicles**: Object detection and traffic monitoring.

---

## **Setup Instructions**

### Prerequisites
- Python 3.11
- AWS account with EC2 instance setup
- GitHub repository with access to CI/CD pipeline

### Steps
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/AbdulRasheed1011/ImageClassification
   cd ImageClassification
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **For Training and Evaluate Model**:
   - Ensure data is placed in the appropriate directory.
   - Update the `params.yaml` with your configuration.
   ```bash
   python main.py
   ```


4. **Deploy the Model on AWS**:
   - Configure GitHub Actions for CI/CD.
   - Push code to the `main` branch to trigger the pipeline.
   - Access the deployed Flask API via the EC2 instance's public IP.

---

## **Project Structure**
```
.
├── source
│   ├── CNNClassifier
│   │   ├── entity
│   │   ├── config
│   │   ├── utils
│   │   └── components
|   |   └── pipeline  
├── artifacts
│   └── data_ingeston
│       └── Images
├── scripts
├── params.yaml
├── requirements.txt
├── main.py
├── dvc.yaml
├── README.md
├── app.py
└── scores.json
```

---

## **Results**
- **Accuracy**: 88.42%
- **Loss**: 0.32
- Model deployed successfully on **AWS EC2** with real-time predictions available via **Flask API**.

---

## **Future Enhancements**
- Expand the dataset for better generalization across diverse categories.
- Implement **distributed training** for faster model optimization.
- Deploy the system using **AWS Lambda** for serverless architecture.
- Add a web interface for user interaction.

---

## **Acknowledgments**
This project utilizes the **VGG16 architecture** and pre-trained weights from **ImageNet**. Special thanks to the open-source community for providing tools and resources to enable efficient development.

