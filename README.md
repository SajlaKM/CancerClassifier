# Brain Cancer Classifier
MRI radionomics application in Brain cancer classification
# Brain Tumor Segmentation and Classification

This project focuses on **brain tumor segmentation and classification** using the **Brain Tumor MRI Dataset** from Kaggle. The dataset contains MRI images labeled into three tumor types: **Glioma, Meningioma, and Pituitary**, along with a **NoTumor** category.

---

## Project Overview  
The project consists of two main tasks:  

1. **Tumor Segmentation** – Using **ResUnet** to segment brain tumors from MRI images.  
2. **Tumor Classification** – Initially implemented with **CNN**, but later improved with **ResNet50**, achieving higher accuracy.  

---

## Dataset  
- **Source:** [Kaggle - Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)  
- The dataset consists of MRI scans categorized into:  
  - **No Tumor**  
  - **Glioma Tumor**  
  - **Meningioma Tumor**  
  - **Pituitary Tumor**  

---

## Project Workflow  

### **Data Preparation**  
- Downloaded the dataset and stored it in **Google Drive**.  
- Mounted the drive and extracted **train** and **test** datasets in Google Colab.  

### **Preprocessing**  
- Normalized MRI images for better model performance.  
- Encoded categorical labels for classification.  
- Created **pseudo masks** for **No Tumor** and **Pituitary Tumor** classes to aid segmentation.  

### **Tumor Segmentation**  
- Used **ResUnet** for tumor segmentation.  
- Generated segmented masks for MRI images.  

### **Tumor Classification**  
- **CNN Model** → Initial classification model (**28% accuracy**).  
- **ResNet50 Model** → Improved classification model (**88% validation accuracy**).  

---

## Results  

| Model   | Accuracy |  
|---------|---------|  
| **CNN** | 28%     |  
| **ResNet50** | 88%  |  

- The **CNN model underperformed**, leading to a transition to **ResNet50**, which significantly improved accuracy.  
- Further enhancements can be made by **fine-tuning ResNet50** or using **EfficientNet** for classification.  

---

## Technologies Used  
 **Python**  
 **Google Colab**  
 **TensorFlow / Keras**  
 **OpenCV**  
 **NumPy, Pandas**  
 **Matplotlib, Seaborn**  

---

## Installation & Usage  

### Clone the Repository  
```bash
git clone https://github.com/your-github-username/brain-tumor-segmentation-classification.git
cd brain-tumor-segmentation-classification
```
### Install Dependencies  
```bash
pip install -r requirements.txt
```
### Run the Notebook in Google Colab  
Upload and open `Brain_Tumor_Segmentation_Classification.ipynb` in Google Colab.  

---

## Future Scope  
✔️ Implement **attention-based U-Net** for better segmentation.  
✔️ Fine-tune **ResNet50** or explore **EfficientNet** for improved classification.  
✔️ Develop a **web app** using **Streamlit** or **Flask** for real-time tumor classification.  

---

## Acknowledgments  
- **Dataset:** [Kaggle - Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)  
- **Model References:** ResUnet, CNN, ResNet50  

---

## Author  
**Sajla KM**  
M.Tech in Artificial Intelligence  
AI/ML Enthusiast | Space Scientist | Data Science & Python Developer  
---

