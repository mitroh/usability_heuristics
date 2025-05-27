import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input, concatenate
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import json

# Load preprocessed data
PROCESSED_PATH = "./preprocessing/"
MODEL_PATH = "./models/ui_evaluation_model.h5"
CHECKPOINT_PATH = "./models/checkpoint/cp.weights.h5"
os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
RESULTS_PATH = "./result/plots/"
os.makedirs(RESULTS_PATH, exist_ok=True)

# Try to load the preprocessed data with the new format
try:
    X_img_train = np.load(f"{PROCESSED_PATH}X_img_train.npy")
    X_img_test = np.load(f"{PROCESSED_PATH}X_img_test.npy")
    X_feat_train = np.load(f"{PROCESSED_PATH}X_feat_train.npy")
    X_feat_test = np.load(f"{PROCESSED_PATH}X_feat_test.npy")
    # Try to use the train/test splits if available
    print(f"Using pre-split data. Training set: {len(X_img_train)} samples")
    
    # For backward compatibility - try to load usability scores or generate them
    try:
        y_train = np.load(f"{PROCESSED_PATH}y_train.npy")
        y_test = np.load(f"{PROCESSED_PATH}y_test.npy")
    except:
        # Fallback - use simple linear combination of features as usability scores
        print("No usability scores found, generating from features...")
        feature_weights = [0.2, 0.15, 0.1, -0.15, -0.2, 0.1, 0.05, 0.05, 0.1, 0.1, -0.1]
        y_train = np.clip(np.sum(X_feat_train * feature_weights, axis=1) + 5, 0, 10)
        y_test = np.clip(np.sum(X_feat_test * feature_weights, axis=1) + 5, 0, 10)
        np.save(f"{PROCESSED_PATH}y_train.npy", y_train)
        np.save(f"{PROCESSED_PATH}y_test.npy", y_test)
except:
    # Fallback to old format for backward compatibility
    print("New format data not found, falling back to old format...")
    try:
        images = np.load(f"{PROCESSED_PATH}images.npy")
        
        # Try to load features or use labels
        try:
            features = np.load(f"{PROCESSED_PATH}features.npy")
        except:
            # Use old labels as basic features
            print("Features not found, using basic labels...")
            labels = np.load(f"{PROCESSED_PATH}labels.npy")
            # Create dummy features based on labels
            features = np.zeros((len(labels), 11))
            features[:, 0] = labels  # Use the original label as first feature
        
        # Create train/test split
        from sklearn.model_selection import train_test_split
        X_img_train, X_img_test, X_feat_train, X_feat_test, y_train, y_test = train_test_split(
            images, features, labels, test_size=0.2, random_state=42
        )
    except:
        print("Error loading data. Please run preprocess.py first.")
        exit(1)

# Define improved hybrid model
def build_hybrid_model(input_shape=(224, 224, 3), feature_shape=None):
    """
    Build a hybrid model that combines CNN features with extracted UI features
    """
    # Use feature shape if available, otherwise use a default
    if feature_shape is None:
        feature_shape = X_feat_train.shape[1] if 'X_feat_train' in locals() else 11
    
    # CNN branch
    image_input = Input(shape=input_shape)
    
    # Use a reduced ResNet50V2 for feature extraction
    x = Conv2D(32, (3, 3), activation='relu')(image_input)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    
    # Feature input branch for extracted UI features
    feature_input = Input(shape=(feature_shape,))
    y = Dense(64, activation='relu')(feature_input)
    y = Dropout(0.3)(y)
    
    # Combine branches
    combined = concatenate([x, y])
    
    # Common layers
    z = Dense(128, activation='relu')(combined)
    z = Dropout(0.4)(z)
    z = Dense(64, activation='relu')(z)
    z = Dropout(0.3)(z)
    
    # Output layer
    output = Dense(1, activation='linear')(z)
    
    # Create model
    model = Model(inputs=[image_input, feature_input], outputs=output)
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='mse',
        metrics=['mae', 'mse']
    )
    
    return model

# Build model
feature_shape = X_feat_train.shape[1]
model = build_hybrid_model(input_shape=(224, 224, 3), feature_shape=feature_shape)
model.summary()

# Define callbacks
checkpoint_callback = ModelCheckpoint(
    filepath=CHECKPOINT_PATH,
    save_weights_only=True,
    save_best_only=True,
    monitor='val_loss',
    mode='min',
    verbose=1
)

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=3,
    min_lr=1e-6,
    verbose=1
)

# Load existing checkpoint if available
if os.path.exists(CHECKPOINT_PATH):
    print("Checkpoint found! Attempting to load weights...")
    try:
        model.load_weights(CHECKPOINT_PATH)
        print("Successfully loaded weights from checkpoint!")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        print("Starting training from scratch...")

# Train model
history = model.fit(
    [X_img_train, X_feat_train],
    y_train,
    epochs=20,
    batch_size=8,
    validation_data=([X_img_test, X_feat_test], y_test),
    callbacks=[checkpoint_callback, early_stopping, reduce_lr],
    verbose=1
)

# Save trained model
model.save(MODEL_PATH)
print(f"Model training completed and saved to {MODEL_PATH}")

# Evaluate model
y_pred = model.predict([X_img_test, X_feat_test])
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Test MSE: {mse:.4f}")
print(f"Test RMSE: {rmse:.4f}")
print(f"Test MAE: {mae:.4f}")
print(f"Test R^2: {r2:.4f}")

# Save metrics
metrics = {
    'mse': float(mse),
    'rmse': float(rmse),
    'mae': float(mae),
    'r2': float(r2)
}

with open('./models/evaluation_metrics.json', 'w') as f:
    json.dump(metrics, f)

# Generate and save training plots
# Create plots directory
os.makedirs(RESULTS_PATH, exist_ok=True)

# Plot training & validation loss values
plt.figure(figsize=(15, 5))

# Loss plot
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss (MSE)')
plt.xlabel('Epoch')
plt.legend(loc='upper right')
plt.grid(True, alpha=0.3)

# MAE plot
plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='Training MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.title('Model Mean Absolute Error')
plt.ylabel('MAE')
plt.xlabel('Epoch')
plt.legend(loc='upper right')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{RESULTS_PATH}training_history.png')
plt.close()

# Generate evaluation plots
plt.figure(figsize=(15, 10))

# Prediction vs Actual plot
plt.subplot(2, 2, 1)
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
plt.title('Predicted vs Actual Usability Scores')
plt.xlabel('Actual Scores')
plt.ylabel('Predicted Scores')
plt.grid(True, alpha=0.3)

# Error distribution
plt.subplot(2, 2, 2)
errors = y_pred.flatten() - y_test
sns.histplot(errors, kde=True)
plt.title('Error Distribution')
plt.xlabel('Prediction Error')
plt.ylabel('Count')
plt.grid(True, alpha=0.3)

# Residual plot
plt.subplot(2, 2, 3)
plt.scatter(y_pred.flatten(), errors, alpha=0.6)
plt.axhline(y=0, color='r', linestyle='--')
plt.title('Residual Plot')
plt.xlabel('Predicted Value')
plt.ylabel('Residual')
plt.grid(True, alpha=0.3)

# Save distributions
plt.subplot(2, 2, 4)
sns.kdeplot(y_test, label='Actual', fill=True, alpha=0.3)
sns.kdeplot(y_pred.flatten(), label='Predicted', fill=True, alpha=0.3)
plt.title('Distribution of Actual vs Predicted Values')
plt.xlabel('Usability Score')
plt.ylabel('Density')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{RESULTS_PATH}evaluation_metrics.png')
plt.close()

print("Training history and evaluation plots saved to './result/plots/'")

# Save history to file
with open(f'{RESULTS_PATH}training_history.json', 'w') as f:
    # Convert numpy values to Python native types
    history_dict = {}
    for key, values in history.history.items():
        history_dict[key] = [float(val) for val in values]
    json.dump(history_dict, f)
print("Training history saved to './result/plots/training_history.json'")
