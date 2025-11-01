# 3D Brain Tumor Segmentation System

![Brain Tumor Segmentation](Screenshot.png)

A deep learning-based web application for automatic segmentation of brain tumors from 3D MRI scans using an attention-based U-Net architecture. This application provides an intuitive web interface for uploading MRI data and visualizing segmentation results in both 3D and 2D views.

## 🎯 Features

### Core Functionality
- **Automatic Brain Tumor Segmentation**: Uses a trained Attention U-Net model to segment brain tumors from MRI scans
- **Multi-Modal MRI Support**: Processes FLAIR, T1CE, and T2 MRI sequences
- **3D Visualization**: Interactive 3D visualization of brain structures and tumor regions using VTK/PyVista
- **2D Slice Visualization**: View MRI data and segmentation masks across multiple slices (X, Y, Z axes)
- **Real-time Inference**: On-the-fly segmentation processing with configurable parameters

### Interactive Controls
- **Modality Selection**: Toggle visibility of different MRI sequences (FLAIR, T1CE, T2)
- **Opacity Adjustment**: Control transparency of brain structures and segmentation masks
- **Threshold Controls**: Adjust visualization thresholds for better rendering
- **Mask Categories**: Visualize different tumor regions (NEC, ED, ET) with color coding
- **Camera Controls**: Reset camera view and interact with 3D models

## 🏗️ Architecture

### Deep Learning Model
- **Model Type**: Attention U-Net (3D)
- **Input Shape**: (192, 192, 3) - 2D slices with 3 MRI modalities
- **Classes**: 4 (Background, Non-Enhancing Tumor, Edema, Enhancing Tumor)
- **Loss Function**: Combined Dice Loss + Focal Loss
- **Preprocessing**: CLAHE (Contrast Limited Adaptive Histogram Equalization)

### Technology Stack
- **Frontend**: Streamlit
- **3D Visualization**: VTK, PyVista, stpyvista
- **Deep Learning**: TensorFlow/Keras
- **Medical Imaging**: Nibabel (NIfTI format support)
- **Image Processing**: OpenCV, NumPy, Matplotlib

## 📦 Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended for faster inference)

### Setup

1. **Clone the repository**
```bash
git clone <repository-url>
cd BraTSModels
```

2. **Create a virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Run the application**
```bash
streamlit run app.py
```

5. **Open your browser**
   Navigate to `http://localhost:8501`

## 📊 Dataset

This project uses the **BraTS (Brain Tumor Segmentation) 2020 Dataset**:

- **Training Data**: 369 multi-modal MRI scans (FLAIR, T1, T1CE, T2)
- **Validation Data**: Separate validation set
- **Data Format**: NIfTI (.nii) files
- **Image Dimensions**: 240 × 240 × 155 voxels
- **Preprocessing**:
  - Cropped to brain region (192 × 192 × 128)
  - Normalized using Min-Max scaling
  - CLAHE enhancement applied
  - Segmentation masks: 4 classes (background, NEC, ED, ET)

### Tumor Classes
- **Class 0**: Background (Healthy tissue)
- **Class 1**: Non-Enhancing Tumor (NEC)
- **Class 2**: Edema (ED)
- **Class 3**: Enhancing Tumor (ET)

## 🚀 Usage

### Step 1: Upload MRI Scans
Upload three MRI sequences in NIfTI format:
- **FLAIR** (Fluid-Attenuated Inversion Recovery)
- **T1CE** (T1-weighted with Contrast Enhancement)
- **T2** (T2-weighted)

### Step 2: Configure Visualization
Adjust parameters in the sidebar:
- **Brain Threshold**: Control brain surface visualization (0.1 - 1.0)
- **Mask Threshold**: Control tumor mask visibility (0.1 - 1.0)
- **Brain Opacity**: Adjust brain transparency (0.1 - 1.0)

### Step 3: Select Modalities
Choose which MRI sequences to display:
- ✅ Show FLAIR (enabled by default)
- ☐ Show T1CE
- ☐ Show T2

### Step 4: Explore Tumor Regions
Visualize different tumor categories:
- Toggle visibility of each tumor class
- Adjust opacity for each region
- Color-coded visualization

### Step 5: 3D Interaction
- **Rotate**: Click and drag
- **Zoom**: Mouse wheel
- **Pan**: Shift + click and drag
- **Reset Camera**: Click "Reset Camera" button

## 📁 Project Structure

```
BraTSModels/
├── app.py                           # Main Streamlit application
├── Attention_Model_Clahe1024.py     # Model training script
├── DataPreprocessing.py             # Data preprocessing pipeline
├── image_loader.py                  # Image loading utilities
├── best_model2.h5                   # Pre-trained model weights
├── requirements.txt                 # Python dependencies
├── .gitignore                       # Git ignore rules
└── README.md                        # Project documentation
```

## 🔬 Technical Details

### Model Architecture
The Attention U-Net incorporates:
- **Encoder-Decoder Structure**: Contracting and expansive paths
- **Skip Connections**: Preserve fine-grained details
- **Attention Gates**: Focus on relevant features
- **Multi-Scale Feature Extraction**: Capture tumors at various scales

### Performance Metrics
The model achieves competitive performance on the BraTS 2020 validation set:
- **Dice Score**: Measures overlap between predicted and ground truth
- **IoU (Intersection over Union)**: Spatial accuracy metric
- **Precision & Recall**: Classification performance
- **F1-Score**: Harmonic mean of precision and recall

### Data Preprocessing Pipeline
1. **Loading**: Load NIfTI files using Nibabel
2. **Normalization**: Min-Max scaling to [0, 1] range
3. **Cropping**: Extract brain region of interest
4. **CLAHE**: Enhance contrast for better feature visibility
5. **Stacking**: Combine 3 modalities into 4D array
6. **Model Input**: 192×192×3 patches for inference

## 🎨 Color Coding

The application uses the following color scheme for segmentation visualization:

- **Red**: Non-Enhancing Tumor (NEC)
- **Blue**: Edema (ED)
- **Green**: Enhancing Tumor (ET)
- **Yellow**: Additional annotations (if present)
- **White**: Brain tissue (MRI modalities)

## 🔧 Configuration Options

### Model Parameters
```python
# Input dimensions
INPUT_SHAPE = (192, 192, 3)
NUM_CLASSES = 4

# Cropping region
START, END = 34, 226
Z_START, Z_END = 13, 141

# CLAHE parameters
CLAHE_CLIP_LIMIT = 2.5
CLAHE_TILE_GRID_SIZE = (9, 9)
```

### Visualization Parameters
```python
# Default thresholds
BRAIN_THRESHOLD = 0.5
MASK_THRESHOLD = 0.5

# Opacity settings
BRAIN_OPACITY = 0.4
MASK_OPACITY = 0.9
```

## 📈 Future Enhancements

- [ ] **Multi-Model Support**: Compare results from different architectures
- [ ] **Evaluation Metrics**: Display quantitative performance metrics
- [ ] **Batch Processing**: Process multiple scans simultaneously
- [ ] **Export Functionality**: Save segmentation masks as NIfTI files
- [ ] **Cloud Deployment**: Deploy on cloud platforms (AWS, GCP, Azure)
- [ ] **Real-time Inference**: Optimize for faster processing
- [ ] **Segmentation Editor**: Manual correction tools
- [ ] **Report Generation**: Automated diagnostic reports

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Guidelines
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **BraTS Community**: For providing the benchmark dataset
- **Medical Imaging Community**: For foundational research in brain tumor segmentation
- **TensorFlow/Keras**: For the deep learning framework
- **PyVista/VTK**: For 3D visualization capabilities

## 📧 Contact

For questions, suggestions, or collaborations, please reach out:

- **Project Link**: [GitHub Repository]
- **Email**: your.email@example.com

## 📚 References

1. **BraTS 2020 Dataset**: [MICCAI BraTS 2020](https://www.med.upenn.edu/cbica/brats2020/)
2. **Attention U-Net**: "Attention U-Net: Learning Where to Look for the Pancreas"
3. **Segmentation Models 3D**: GitHub repository for 3D segmentation architectures

---

**Note**: This application is intended for research and educational purposes only. It should not be used for clinical diagnosis without proper validation and regulatory approval.
