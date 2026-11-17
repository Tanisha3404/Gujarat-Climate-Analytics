import pandas as pd
import numpy as np
import pickle
import os
import warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" 

from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (LSTM, Dense, Dropout,
                                     Conv1D, MaxPooling1D, Flatten)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

DATA_FOLDER   = r"C:\Users\HP\Desktop\DA\Data"
PREP_FOLDER   = os.path.join(DATA_FOLDER, "preprocessed")
OUTPUT_FOLDER = r"C:\Users\HP\Desktop\DA\models\models_results"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

SEQ_LEN = 30

print("=" * 60)
print("STEP 4d: LSTM Temperature Forecasting")
print("=" * 60)
print(f"\n  TensorFlow version : {tf.__version__}")
gpu = tf.config.list_physical_devices("GPU")
print(f"  GPU available      : {'Yes — ' + gpu[0].name if gpu else 'No — using CPU (slower but fine)'}")

# Load LSTM preprocessed data 
with open(os.path.join(PREP_FOLDER, "lstm_data.pkl"), "rb") as f:
    lstm_data = pickle.load(f)

cities = list(lstm_data.keys())
print(f"  Cities to train    : {cities}")

# Model builders 
def build_lstm(seq_len, n_features):
    """Standard 2-layer LSTM."""
    model = Sequential([
        LSTM(64, return_sequences=True,
             input_shape=(seq_len, n_features)),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(1),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss="mse", metrics=["mae"])
    return model

def build_cnn_lstm(seq_len, n_features):
    """CNN-LSTM hybrid — CNN extracts local patterns, LSTM captures long-range."""
    model = Sequential([
        Conv1D(filters=64, kernel_size=3, activation="relu",
               input_shape=(seq_len, n_features)),
        MaxPooling1D(pool_size=2),
        Conv1D(filters=32, kernel_size=3, activation="relu"),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(1),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss="mse", metrics=["mae"])
    return model

# Callbacks 
def get_callbacks():
    return [
        EarlyStopping(monitor="val_loss", patience=10,
                      restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                          patience=5, min_lr=1e-6, verbose=0),
    ]

#Train per city 
all_results = {}

for city in cities:
    print(f"\n{'='*60}")
    print(f"  Training: {city}")
    print(f"{'='*60}")

    city_data = lstm_data[city]
    X_train = city_data["X_train"]
    y_train = city_data["y_train"]
    X_test  = city_data["X_test"]
    y_test  = city_data["y_test"]
    scaler  = city_data["scaler"]

    n_features = X_train.shape[2]
    print(f"  X_train: {X_train.shape}  X_test: {X_test.shape}")

    #Train LSTM 
    print(f"\n  Training LSTM...")
    lstm_model = build_lstm(SEQ_LEN, n_features)

    history_lstm = lstm_model.fit(
        X_train, y_train,
        validation_split=0.1,
        epochs=100,
        batch_size=32,
        callbacks=get_callbacks(),
        verbose=1,
    )

    # Train CNN-LSTM
    print(f"\n  Training CNN-LSTM...")
    cnn_lstm_model = build_cnn_lstm(SEQ_LEN, n_features)

    history_cnn = cnn_lstm_model.fit(
        X_train, y_train,
        validation_split=0.1,
        epochs=100,
        batch_size=32,
        callbacks=get_callbacks(),
        verbose=1,
    )

    #Evaluate both models
    def evaluate(model, X_test, y_test, scaler, label):
        y_pred_scaled = model.predict(X_test, verbose=0).flatten()

        # Inverse transform — we need to rebuild the full feature array
        # since scaler was fit on all features together
        # We place predictions in the avg_temp column (index 0) and inverse
        n_feat = scaler.n_features_in_
        dummy  = np.zeros((len(y_pred_scaled), n_feat))
        dummy[:, 0] = y_pred_scaled
        y_pred = scaler.inverse_transform(dummy)[:, 0]

        dummy2 = np.zeros((len(y_test), n_feat))
        dummy2[:, 0] = y_test
        y_true = scaler.inverse_transform(dummy2)[:, 0]

        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae  = mean_absolute_error(y_true, y_pred)
        r2   = 1 - np.sum((y_true-y_pred)**2) / np.sum((y_true-y_true.mean())**2)

        print(f"  {label:15s}  RMSE={rmse:.4f}°C  MAE={mae:.4f}°C  R²={r2:.4f}")
        return {"rmse": rmse, "mae": mae, "r2": r2,
                "y_true": y_true, "y_pred": y_pred}

    print(f"\n  Results for {city}:")
    lstm_eval    = evaluate(lstm_model,     X_test, y_test, scaler, "LSTM")
    cnn_lstm_eval = evaluate(cnn_lstm_model, X_test, y_test, scaler, "CNN-LSTM")

    # Best model for this city
    best_label = "LSTM" if lstm_eval["rmse"] <= cnn_lstm_eval["rmse"] else "CNN-LSTM"
    best_rmse  = min(lstm_eval["rmse"], cnn_lstm_eval["rmse"])
    print(f"  Best model: {best_label}  (RMSE={best_rmse:.4f}°C)")

    all_results[city] = {
        "lstm_model":     lstm_model,
        "cnn_lstm_model": cnn_lstm_model,
        "lstm_eval":      lstm_eval,
        "cnn_lstm_eval":  cnn_lstm_eval,
        "history_lstm":   history_lstm.history,
        "history_cnn":    history_cnn.history,
        "scaler":         scaler,
        "best_model":     best_label,
    }

    # Save individual city model weights
    lstm_model.save(os.path.join(OUTPUT_FOLDER, f"lstm_{city.lower()}.keras"))
    cnn_lstm_model.save(os.path.join(OUTPUT_FOLDER, f"cnn_lstm_{city.lower()}.keras"))
    print(f"  Models saved for {city}")

#Final summary across all cities
print(f"\nLSTM FINAL SUMMARY — All Cities")
print(f"\n  {'City':12s} {'LSTM RMSE':>12} {'LSTM R²':>10} {'CNN-LSTM RMSE':>14} {'CNN-LSTM R²':>12} {'Best':>8}")
print(f"  {'─'*72}")
for city, res in all_results.items():
    le = res["lstm_eval"];  ce = res["cnn_lstm_eval"]
    print(f"  {city:12s} {le['rmse']:>12.4f} {le['r2']:>10.4f} "
          f"{ce['rmse']:>14.4f} {ce['r2']:>12.4f} {res['best_model']:>8}")


save_path = os.path.join(OUTPUT_FOLDER, "lstm_results.pkl")
with open(save_path, "wb") as f:
    pickle.dump(all_results, f)

print(f"\n  Saved → {save_path}")
