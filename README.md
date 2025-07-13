# AI-Based Multimodal Deception Detection using DOLOS Dataset

This project implements a deep learning-based **multimodal deception detection system** that analyzes both **audio and video inputs** to classify human responses as **truthful or deceptive**. The system integrates **ResNet18**, **LSTM**, **DeepFace**, and **NLP techniques (TF-IDF, Word2Vec)** for a robust and interpretable prediction.

---

## 🧠 Features

- ResNet18 + LSTM hybrid model for multimodal fusion
- Audio spectrogram and frame analysis
- Facial emotion detection using DeepFace
- NLP-based classification using TF-IDF and Word2Vec
- SHAP and LIME support for explainability
- Streamlit-powered web interface

---

## 🚀 How to Run

```bash
# Step 1: Install required dependencies
pip install -r requirements.txt

# Step 2: Launch the web app
streamlit run app.py
```

> ⚠️ The model file (`lstmfusionvm.pth`) is **private** and not included in this repository.  
> Without this file, the prediction module will be disabled.  

---

## 📁 Dataset

This project uses the [DOLOS dataset](https://github.com/MahmudulAlam/Audio-Visual-Deception-Detection-DOLOS-Dataset-and-Parameter-Efficient-Crossmodal-Learning), which is not shared due to license restrictions.  
Download the dataset manually and organize it in a `dataset/` directory as follows:

```
dataset/
├── truthful/
│   ├── sample1.mp4
│   └── ...
├── deceptive/
│   ├── sample1.mp4
│   └── ...
```

---

## 📜 License & Citation

This project is licensed under the **GNU GPLv3 License** and **Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International** 
Commercial use and redistribution of the model or components without explicit permission is **prohibited**.

> 📣 **Citation Required**:  
> If you use any part of this project in your work (code, design, or idea), **you must cite this repository**.  

Example BibTeX:

```bibtex
@misc{ai_deception_detection,
  author = {Varun},
  title = {AI-Based Multimodal Deception Detection using DOLOS Dataset},
  year = {2025},
  howpublished = {GitHub},
  url = {https://github.com/your-username/ai-deception-detection}
}
```

---

## 🙋 Acknowledgements

- DOLOS Dataset by Mahmudul Alam et al.
- DeepFace (Facial Analysis)
- Torch, OpenCV, Streamlit, Scikit-learn, SHAP, and LIME

