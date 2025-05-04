# LeNet-5_Classifier
This task explores the training and optimization of a Convolutional Neural Network (CNN) based on the LeNet-5 architecture for classifying handwritten digits from the ReducedMNIST dataset. The study evaluates the impact of various hyperparameters and modifications on model performance.
---
## Architecture (parameters have been modified for 28x28 images)
![image](https://github.com/user-attachments/assets/e102a678-8973-4f3a-8d57-803cd2b10289)

## Key Experiments and Results

### 1. **Initial LeNet-5 CNN**
- **Configuration**:  
  - Mini-batch size: 32  
  - Learning rate: 0.001  
  - Pooling: Average pooling  
  - Epochs: 5  
- **Results**:  
  - Training Accuracy: 97.7%  
  - Testing Accuracy: 97.4%  
  - Training Time: 23,951.9 ms  

### 2. **CNN Variants**
Four modifications were tested to improve performance:

#### **3.1 Mini-Batch Size Reduction**
- **Change**: Reduced from 32 to 8.  
- **Impact**:  
  - Improved accuracy (Training: 98.8%, Testing: 97.8%).  
  - Doubled training time due to more frequent updates.  

#### **3.2 Learning Rate Adjustment**
- **Change**: Increased from 0.001 to 0.003.  
- **Impact**:  
  - Faster convergence and slight accuracy boost (Training: 98.5%, Testing: 98%).  
  - Marginal increase in training time.  

#### **3.3 Pooling Layer Modification**
- **Change**: Replaced average pooling with max pooling.  
- **Impact**:  
  - Better preservation of key features (Training: 98.9%, Testing: 98.1%).  
  - Slightly longer training time.  

#### **3.4 Batch Normalization**
- **Change**: Added after each pooling layer (inspired by AlexNet).  
- **Impact**:  
  - Highest accuracy (Training: 99.1%, Testing: 98%).  
  - Minimal impact on training time.  

---

## Performance Comparison

| **Model/Variant**               | **Training Accuracy (%)** | **Testing Accuracy (%)** | **Training Time (ms)** |
|---------------------------------|--------------------------|--------------------------|------------------------|
| LeNet-5 (Baseline)              | 97.7                     | 97.4                     | 23,951.9               |
| Mini-Batch Size = 8             | 98.8                     | 97.8                     | 38,424.7               |
| Max Pooling                     | 98.9                     | 98.1                     | 26,202.3               |
| Learning Rate = 0.003           | 98.5                     | 98                        | 25,365.8               |
| Batch Normalization             | 99.1                     | 98                        | 24,057.2               |

---

## Conclusions
1. **CNN Superiority**: CNNs outperformed classical methods (e.g., SVM, K-means) and MLPs in accuracy, achieving up to **99.1% training accuracy** and **98% testing accuracy**.  
2. **Trade-offs**: Modifications like smaller mini-batches and max pooling improved accuracy but increased computational cost.  
3. **Optimal Configuration**: Batch normalization and adjusted learning rates offered the best balance between accuracy and training efficiency.  
4. **Generalization**: All variants maintained close training-testing accuracy gaps, indicating robust generalization.  

---

