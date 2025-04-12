# American Sign Language Recognition System

A deep learning-based system that recognizes and classifies American Sign Language (ASL) gestures in real-time using video input. Designed to aid non-verbal communication by providing an intuitive, accessible interface powered by computer vision and neural networks.

---

## Features

- **Real-time gesture detection** using OpenCV and live video input
- **CNN-based model** trained on ASL hand gesture dataset with over 95% accuracy
- **24 ASL alphabets** supported (A-Z excluding J and Z due to motion)
- Lightweight interface built for accessibility and ease of use
- Modular codebase allowing easy extension to other gesture-based systems

---

## Tech Stack

- **Programming Language**: Python  
- **Libraries & Frameworks**:
  - TensorFlow / Keras
  - OpenCV
  - NumPy
  - Matplotlib
  - Scikit-learn
  -  Pillow (PIL)
  - seaborn

---

## Demo

![ASL Demo](demo/demo.gif)  
*Live detection of ASL gestures via webcam*


---

## Dataset

- Used the [American Sign Language Letters dataset](https://www.kaggle.com/datasets/danrasband/asl-alphabet-test) from Kaggle.
- Includes over 87,000 images of 29 classes (A-Z, nothing, space, delete).
- Only 24 static letter classes were used in this model.

---

## Model Architecture

- Convolutional Neural Network (CNN)
  - 3 Convolutional Layers + ReLU + MaxPooling
  - Fully Connected Layer
  - Dropout for regularization
  - Softmax for multiclass classification

Training was done using TensorFlow/Keras with early stopping and model checkpointing to prevent overfitting.

---

## How to Run

### 1. Clone the repository
```bash
git clone https://github.com/YourUsername/asl-recognition.git
cd asl-recognition
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Train the model (optional)
```bash
python train.py
```

### 4. Run live ASL prediction
```bash
python predict_live.py
```

---

## File Structure

```
asl-recognition/
├── model/                 # Saved trained model
├── data/                  # Preprocessed dataset
├── predict_live.py        # Webcam-based real-time detection
├── train.py               # Model training script
├── utils.py               # Helper functions
├── demo/                  # Demo gifs/videos/images
└── README.md
```

---

## Future Improvements

- Add support for motion-based letters (J and Z)
- Include sentence-level gesture recognition
- Integrate audio feedback via TTS for accessibility

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Contact

**Ashmit Tatia**  
[LinkedIn](https://www.linkedin.com/in/ashmit-tatia-367b6624b/)
