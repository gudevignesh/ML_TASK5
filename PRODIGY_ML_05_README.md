# 🍕 PRODIGY_ML_05 — Food Recognition & Calorie Estimation

> **Prodigy InfoTech Machine Learning Internship — Task 05**

## 📌 Task Description
Develop a model that can accurately recognize food items from images and estimate their calorie content, enabling users to track their dietary intake and make informed food choices.

---

## 📂 Dataset
**Food-101 — Kaggle**  
🔗 https://www.kaggle.com/dansbecker/food-101

- 101 food categories
- 1,000 images per category
- 101,000 total images

---

## 🛠️ Tech Stack
| Tool | Purpose |
|------|---------|
| Python 3.x | Programming Language |
| TensorFlow/Keras | Deep Learning Framework |
| MobileNetV2 | Pre-trained Model (Transfer Learning) |
| OpenCV (cv2) | Image preprocessing |
| Matplotlib | Visualizations |
| Seaborn | Statistical plots |
| NumPy | Numerical computations |

---

## 📁 Project Structure
```
PRODIGY_ML_05/
│
├── food_recognition_calories.py  # Main Python script
├── sample_foods.png              # Sample food images
├── training_history.png          # Training accuracy & loss
├── confusion_matrix.png          # Model confusion matrix
├── calorie_predictions.png       # Food predictions with calories
├── calorie_chart.png             # Calorie comparison chart
└── README.md                     # Project documentation
```

---

## ⚙️ How to Run on Google Colab

### 1. Download Dataset
```python
import os
os.makedirs('/root/.kaggle', exist_ok=True)
with open('/root/.kaggle/kaggle.json', 'w') as f:
    f.write('{"username":"YOUR_USERNAME","key":"YOUR_API_KEY"}')
os.chmod('/root/.kaggle/kaggle.json', 0o600)
!pip install kaggle -q
!kaggle datasets download -d dansbecker/food-101
!unzip food-101.zip -d food101
```

### 2. Run the script
- Enable GPU: Runtime → Change runtime type → T4 GPU
- Paste code from `food_recognition_calories.py`
- Click Run and wait for results!

---

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| Model | MobileNetV2 (Transfer Learning) |
| Food Classes | 10 |
| Training Images | 1,600 |
| Testing Images | 400 |
| **Test Accuracy** | **78.25%** |

---

## 🍽️ Selected Food Classes & Calories

| Food Item | Calories/100g |
|-----------|--------------|
| 🍕 Pizza | 266 |
| 🍔 Hamburger | 295 |
| 🍣 Sushi | 150 |
| 🌮 Tacos | 226 |
| 🍦 Ice Cream | 207 |
| 🍩 Donuts | 452 |
| 🍟 French Fries | 312 |
| 🍫 Chocolate Cake | 371 |
| 🥞 Pancakes | 227 |
| 🍜 Ramen | 65 |

---

## 📈 Key Visualizations
- **Sample Foods** — Sample images from each food class with calories
- **Training History** — Accuracy and loss over epochs
- **Confusion Matrix** — Detailed prediction accuracy per class
- **Calorie Predictions** — Food recognition with calorie estimation
- **Calorie Chart** — Visual comparison of calories across food items

---

## 🔍 Key Findings
- MobileNetV2 Transfer Learning achieved **78.25% accuracy**
- Donuts have the highest calories (452 cal/100g)
- Ramen has the lowest calories (65 cal/100g)
- Transfer Learning is much faster than training from scratch

---

## 👤 Author
**Umesh**  
Machine Learning Intern — Prodigy InfoTech  
🔗 [GitHub](https://github.com/Umesh3516)

---

## 📜 License
This project is for educational purposes as part of an internship program.
