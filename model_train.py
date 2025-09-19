# model_train.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow import keras
import joblib

# 1. Load dataset
df = pd.read_csv("hand_data.csv")

# 2. Split features (X) and labels (y)
X = df.drop("label", axis=1).values
y = df["label"].values

# 3. Encode labels into integers
le = LabelEncoder()
y_int = le.fit_transform(y)   # "fist" -> 0, "open" -> 1, etc.

# 4. Convert to one-hot
num_classes = len(le.classes_)
y_cat = keras.utils.to_categorical(y_int, num_classes)

# 5. Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save scaler + label encoder classes
joblib.dump(scaler, "scaler.pkl")
np.save("classes.npy", le.classes_)

# 6. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_cat, test_size=0.2, random_state=42
)

# 7. Build model
model = keras.Sequential([
    keras.layers.Input(shape=(X.shape[1],)),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(num_classes, activation="softmax")
])

model.compile(optimizer="adam",
              loss="categorical_crossentropy",
              metrics=["accuracy"])

# 8. Train
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=20,
    batch_size=32,
    verbose=1
)

# 9. Save model
model.save("gesture_model.keras")   # modern format
model.save("gesture_model.h5")      # legacy format
print("✅ Model + scaler + classes saved.")

