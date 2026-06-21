# ============================================================
# PRODIGY INFOTECH - Machine Learning Internship
# Task 05: Food Recognition & Calorie Estimation
# Dataset: Food-101 (Kaggle)
# Platform: Google Colab
# ============================================================

# ─────────────────────────────────────────
# 1. INSTALL & IMPORT LIBRARIES
# ─────────────────────────────────────────
!pip install tensorflow -q

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

print("=" * 55)
print("  PRODIGY INFOTECH - ML Task 05")
print("  Food Recognition & Calorie Estimation")
print("=" * 55)
print(f"\n✅ TensorFlow version: {tf.__version__}")
print("✅ All libraries imported successfully!")

# ─────────────────────────────────────────
# 2. CALORIE DATABASE
# ─────────────────────────────────────────
# Approximate calories per 100g for selected food items
CALORIE_DB = {
    'apple_pie'         : 237,
    'baby_back_ribs'    : 292,
    'baklava'           : 428,
    'beef_carpaccio'    : 135,
    'beef_tartare'      : 196,
    'beet_salad'        : 45,
    'beignets'          : 320,
    'bibimbap'          : 120,
    'bread_pudding'     : 153,
    'breakfast_burrito' : 191,
    'bruschetta'        : 195,
    'caesar_salad'      : 90,
    'cannoli'           : 310,
    'caprese_salad'     : 95,
    'carrot_cake'       : 415,
    'ceviche'           : 80,
    'cheesecake'        : 321,
    'cheese_plate'      : 350,
    'chicken_curry'     : 150,
    'chicken_quesadilla': 220,
    'chicken_wings'     : 290,
    'chocolate_cake'    : 371,
    'chocolate_mousse'  : 180,
    'churros'           : 375,
    'clam_chowder'      : 80,
    'club_sandwich'     : 290,
    'crab_cakes'        : 190,
    'creme_brulee'      : 185,
    'croque_madame'     : 270,
    'cup_cakes'         : 305,
    'deviled_eggs'      : 145,
    'donuts'            : 452,
    'dumplings'         : 175,
    'edamame'           : 121,
    'eggs_benedict'     : 250,
    'escargots'         : 90,
    'falafel'           : 333,
    'filet_mignon'      : 267,
    'fish_and_chips'    : 295,
    'foie_gras'         : 462,
    'french_fries'      : 312,
    'french_onion_soup' : 57,
    'french_toast'      : 166,
    'fried_calamari'    : 175,
    'fried_rice'        : 163,
    'frozen_yogurt'     : 127,
    'garlic_bread'      : 350,
    'gnocchi'           : 130,
    'greek_salad'       : 75,
    'grilled_cheese_sandwich': 380,
    'grilled_salmon'    : 206,
    'guacamole'         : 150,
    'gyoza'             : 210,
    'hamburger'         : 295,
    'hot_and_sour_soup' : 45,
    'hot_dog'           : 290,
    'huevos_rancheros'  : 180,
    'hummus'            : 177,
    'ice_cream'         : 207,
    'lasagna'           : 135,
    'lobster_bisque'    : 95,
    'lobster_roll_sandwich': 285,
    'macaroni_and_cheese': 164,
    'macarons'          : 420,
    'miso_soup'         : 40,
    'mussels'           : 86,
    'nachos'            : 306,
    'omelette'          : 154,
    'onion_rings'       : 411,
    'oysters'           : 69,
    'pad_thai'          : 160,
    'paella'            : 180,
    'pancakes'          : 227,
    'panna_cotta'       : 168,
    'peking_duck'       : 337,
    'pho'               : 55,
    'pizza'             : 266,
    'pork_chop'         : 231,
    'poutine'           : 260,
    'prime_rib'         : 300,
    'pulled_pork_sandwich': 280,
    'ramen'             : 65,
    'ravioli'           : 220,
    'red_velvet_cake'   : 366,
    'risotto'           : 166,
    'samosa'            : 262,
    'sashimi'           : 127,
    'scallops'          : 111,
    'seaweed_salad'     : 45,
    'shrimp_and_grits'  : 180,
    'spaghetti_bolognese': 163,
    'spaghetti_carbonara': 200,
    'spring_rolls'      : 165,
    'steak'             : 271,
    'strawberry_shortcake': 245,
    'sushi'             : 150,
    'tacos'             : 226,
    'takoyaki'          : 195,
    'tiramisu'          : 240,
    'tuna_tartare'      : 130,
    'waffles'           : 291,
}

print(f"\n🍽️ Calorie database loaded: {len(CALORIE_DB)} food items")

# ─────────────────────────────────────────
# 3. SELECT FOOD CLASSES (Top 10 for speed)
# ─────────────────────────────────────────
SELECTED_CLASSES = [
    'pizza', 'hamburger', 'sushi', 'tacos',
    'ice_cream', 'donuts', 'french_fries',
    'chocolate_cake', 'pancakes', 'ramen'
]

print(f"\n🍕 Selected {len(SELECTED_CLASSES)} food classes:")
for i, food in enumerate(SELECTED_CLASSES):
    cal = CALORIE_DB.get(food, 'N/A')
    print(f"   {i+1:2d}. {food:<25} → {cal} cal/100g")

# ─────────────────────────────────────────
# 4. SETUP DATA DIRECTORIES
# ─────────────────────────────────────────
base_path  = 'food101/food-101/images'
train_path = '/tmp/food_train'
test_path  = '/tmp/food_test'

os.makedirs(train_path, exist_ok=True)
os.makedirs(test_path,  exist_ok=True)

import shutil
import random

MAX_PER_CLASS = 200  # 200 images per class for faster training

print(f"\n📁 Organizing dataset...")
print(f"   Max per class: {MAX_PER_CLASS}")

for food_class in SELECTED_CLASSES:
    src_dir   = os.path.join(base_path, food_class)
    train_dir = os.path.join(train_path, food_class)
    test_dir  = os.path.join(test_path,  food_class)

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir,  exist_ok=True)

    images = [f for f in os.listdir(src_dir)
              if f.endswith('.jpg')][:MAX_PER_CLASS]
    random.shuffle(images)

    split     = int(0.8 * len(images))
    train_imgs = images[:split]
    test_imgs  = images[split:]

    for img in train_imgs:
        shutil.copy(os.path.join(src_dir, img),
                    os.path.join(train_dir, img))
    for img in test_imgs:
        shutil.copy(os.path.join(src_dir, img),
                    os.path.join(test_dir, img))

    print(f"   ✅ {food_class:<25}: {len(train_imgs)} train, {len(test_imgs)} test")

# ─────────────────────────────────────────
# 5. DATA GENERATORS
# ─────────────────────────────────────────
IMG_SIZE   = 224
BATCH_SIZE = 32

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_path,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

class_names = list(train_generator.class_indices.keys())
print(f"\n✅ Data generators ready!")
print(f"   Training samples  : {train_generator.samples}")
print(f"   Testing samples   : {test_generator.samples}")
print(f"   Classes           : {class_names}")

# ─────────────────────────────────────────
# 6. EDA — SAMPLE IMAGES
# ─────────────────────────────────────────
fig, axes = plt.subplots(2, 5, figsize=(18, 8))
fig.suptitle('Prodigy InfoTech — Task 05: Sample Food Images',
             fontsize=14, fontweight='bold', color='#3a0ca3')

for i, food in enumerate(SELECTED_CLASSES):
    row = i // 5
    col = i % 5
    food_dir = os.path.join(train_path, food)
    img_file = os.listdir(food_dir)[0]
    img      = cv2.imread(os.path.join(food_dir, img_file))
    img      = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img      = cv2.resize(img, (224, 224))
    axes[row, col].imshow(img)
    cal = CALORIE_DB.get(food, 'N/A')
    axes[row, col].set_title(f'{food}\n{cal} cal/100g',
                              fontsize=8, fontweight='bold')
    axes[row, col].axis('off')

plt.tight_layout()
plt.savefig('sample_foods.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n✅ Sample foods plot saved!")

# ─────────────────────────────────────────
# 7. BUILD MODEL (MobileNetV2 Transfer Learning)
# ─────────────────────────────────────────
print("\n🤖 Building MobileNetV2 model...")

base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)
base_model.trainable = False  # Freeze base model

# Add custom layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(len(SELECTED_CLASSES), activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print(f"✅ Model built successfully!")
print(f"   Base model  : MobileNetV2 (ImageNet weights)")
print(f"   Output classes: {len(SELECTED_CLASSES)}")
model.summary()

# ─────────────────────────────────────────
# 8. TRAIN MODEL
# ─────────────────────────────────────────
print("\n🏋️ Training model (5-10 minutes)...")

callbacks = [
    EarlyStopping(patience=3, restore_best_weights=True),
    ReduceLROnPlateau(factor=0.5, patience=2)
]

history = model.fit(
    train_generator,
    epochs=10,
    validation_data=test_generator,
    callbacks=callbacks,
    verbose=1
)
print("\n✅ Model trained successfully!")

# ─────────────────────────────────────────
# 9. PLOT TRAINING HISTORY
# ─────────────────────────────────────────
fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
fig2.suptitle('Prodigy InfoTech — Task 05: Training History',
              fontsize=14, fontweight='bold', color='#3a0ca3')

axes2[0].plot(history.history['accuracy'],
              color='#4361ee', linewidth=2, label='Train')
axes2[0].plot(history.history['val_accuracy'],
              color='#f72585', linewidth=2, label='Validation')
axes2[0].set_title('Model Accuracy')
axes2[0].set_xlabel('Epoch')
axes2[0].set_ylabel('Accuracy')
axes2[0].legend()
axes2[0].grid(True, alpha=0.3)

axes2[1].plot(history.history['loss'],
              color='#4361ee', linewidth=2, label='Train')
axes2[1].plot(history.history['val_loss'],
              color='#f72585', linewidth=2, label='Validation')
axes2[1].set_title('Model Loss')
axes2[1].set_xlabel('Epoch')
axes2[1].set_ylabel('Loss')
axes2[1].legend()
axes2[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n✅ Training history plot saved!")

# ─────────────────────────────────────────
# 10. EVALUATE MODEL
# ─────────────────────────────────────────
print("\n📊 Evaluating model...")
test_loss, test_accuracy = model.evaluate(test_generator, verbose=0)

print("\n" + "=" * 45)
print("  📊 MODEL PERFORMANCE")
print("=" * 45)
print(f"  Test Accuracy : {test_accuracy*100:.2f}%")
print(f"  Test Loss     : {test_loss:.4f}")
print("=" * 45)

# ─────────────────────────────────────────
# 11. CONFUSION MATRIX
# ─────────────────────────────────────────
y_pred = model.predict(test_generator, verbose=0)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true         = test_generator.classes

fig3, ax3 = plt.subplots(figsize=(12, 10))
cm = confusion_matrix(y_true, y_pred_classes)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names, ax=ax3)
ax3.set_title(f'Prodigy InfoTech — Task 05: Confusion Matrix\nAccuracy: {test_accuracy*100:.2f}%',
              fontsize=14, fontweight='bold', color='#3a0ca3')
ax3.set_xlabel('Predicted')
ax3.set_ylabel('Actual')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n✅ Confusion matrix saved!")

# ─────────────────────────────────────────
# 12. CALORIE PREDICTION VISUALIZATION
# ─────────────────────────────────────────
fig4, axes4 = plt.subplots(2, 5, figsize=(18, 9))
fig4.suptitle('Prodigy InfoTech — Task 05: Food Recognition & Calorie Estimation',
              fontsize=13, fontweight='bold', color='#3a0ca3')

test_images_dir = test_path
sample_results  = []

for i, food in enumerate(SELECTED_CLASSES):
    row = i // 5
    col = i % 5
    food_dir  = os.path.join(test_path, food)
    img_files = os.listdir(food_dir)
    if not img_files:
        continue

    img_file = img_files[0]
    img_path = os.path.join(food_dir, img_file)
    img      = cv2.imread(img_path)
    img_rgb  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_disp = cv2.resize(img_rgb, (224, 224))

    # Predict
    img_input = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
    img_input = img_input / 255.0
    img_input = np.expand_dims(img_input, axis=0)
    pred      = model.predict(img_input, verbose=0)
    pred_idx  = np.argmax(pred)
    pred_food = class_names[pred_idx]
    confidence = pred[0][pred_idx] * 100
    calories   = CALORIE_DB.get(pred_food, 'N/A')

    correct = pred_food == food
    color   = 'green' if correct else 'red'

    axes4[row, col].imshow(img_disp)
    axes4[row, col].set_title(
        f'Actual: {food}\nPred: {pred_food}\n'
        f'Cal: {calories}/100g ({confidence:.1f}%)',
        color=color, fontsize=7
    )
    axes4[row, col].axis('off')

plt.tight_layout()
plt.savefig('calorie_predictions.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Calorie predictions plot saved!")

# ─────────────────────────────────────────
# 13. CALORIE CHART
# ─────────────────────────────────────────
fig5, ax5 = plt.subplots(figsize=(14, 6))
foods    = SELECTED_CLASSES
calories = [CALORIE_DB.get(f, 0) for f in foods]
colors   = ['#f72585' if c > 300 else '#4361ee' if c > 150
            else '#4cc9f0' for c in calories]

bars = ax5.bar(foods, calories, color=colors,
               edgecolor='white', linewidth=1.5)
ax5.set_title('Prodigy InfoTech — Task 05: Calories per 100g',
              fontweight='bold', color='#3a0ca3', fontsize=13)
ax5.set_xlabel('Food Item')
ax5.set_ylabel('Calories per 100g')
ax5.tick_params(axis='x', rotation=45)

for bar, cal in zip(bars, calories):
    ax5.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 3,
             str(cal), ha='center',
             fontweight='bold', fontsize=9)

# Legend
from matplotlib.patches import Patch
legend = [
    Patch(color='#f72585', label='High (>300 cal)'),
    Patch(color='#4361ee', label='Medium (150-300 cal)'),
    Patch(color='#4cc9f0', label='Low (<150 cal)')
]
ax5.legend(handles=legend)
plt.tight_layout()
plt.savefig('calorie_chart.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Calorie chart saved!")

# ─────────────────────────────────────────
# 14. DOWNLOAD OUTPUT FILES
# ─────────────────────────────────────────
from google.colab import files
print("\n📥 Downloading output files...")
files.download('sample_foods.png')
files.download('training_history.png')
files.download('confusion_matrix.png')
files.download('calorie_predictions.png')
files.download('calorie_chart.png')

# ─────────────────────────────────────────
# 15. FINAL SUMMARY
# ─────────────────────────────────────────
print("\n" + "=" * 55)
print("  📊 FINAL SUMMARY")
print("=" * 55)
print(f"  Model          : MobileNetV2 (Transfer Learning)")
print(f"  Food Classes   : {len(SELECTED_CLASSES)}")
print(f"  Training Images: {train_generator.samples}")
print(f"  Testing Images : {test_generator.samples}")
print(f"  Test Accuracy  : {test_accuracy*100:.2f}%")
print("=" * 55)
print("\n✅ Task 05 Complete!")
print("   → sample_foods.png")
print("   → training_history.png")
print("   → confusion_matrix.png")
print("   → calorie_predictions.png")
print("   → calorie_chart.png")
