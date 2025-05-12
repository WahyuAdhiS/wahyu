import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from datetime import timedelta
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import math
from tensorflow.keras.callbacks import Callback


# Tambahkan Time2Vec untuk irregular timesteps
class Time2Vec(layers.Layer):
    def __init__(self, output_dims):
        super().__init__()
        self.output_dims = output_dims
        self.w = layers.Dense(1)
        self.p = layers.Dense(output_dims-1, activation='sin')
    
    def call(self, t):
        return tf.concat([self.w(t), self.p(t)], axis=-1)
        
# Cosine Annealing Scheduler for learning rate adjustment
class CosineAnnealingScheduler(Callback):
    def __init__(self, max_lr, min_lr, epochs, verbose=0):
        super().__init__()
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.epochs = epochs
        self.verbose = verbose

    def on_epoch_begin(self, epoch, logs=None):
        cosine_decay = 0.5 * (1 + math.cos(math.pi * epoch / self.epochs))
        new_lr = self.min_lr + (self.max_lr - self.min_lr) * cosine_decay
        tf.keras.backend.set_value(self.model.optimizer.lr, new_lr)
        if self.verbose:
            print(f"\nEpoch {epoch+1}: Cosine Annealing LR = {new_lr:.6f}")


# Variable Selection Layer (VSN)
class VariableSelectionLayer(layers.Layer):
    def __init__(self, input_dim, units=64):
        super(VariableSelectionLayer, self).__init__()
        self.dense = layers.Dense(units, activation='sigmoid')
        self.selection_weights = layers.Dense(input_dim, activation='sigmoid')

    def call(self, inputs):
        weights = self.selection_weights(inputs)
        return inputs * weights  # Seleksi fitur berdasarkan bobot yang dipelajari


# TFT Stock Forecaster class
class TFTStockForecaster:
    def __init__(self, file_path='List/S_ARTO.xlsx', ihsg_path='List/INDEX.xlsx'):
        self.file_path = file_path
        self.ihsg_path = ihsg_path
        self.scaler = MinMaxScaler()
        self.model = None
        self.sequence_length = 100
        self.original_df = None
        self.quantiles = [0.1, 0.5, 0.9]

    def add_positional_encoding(self, x):
        seq_len = tf.shape(x)[1]  # Sequence length
        d_model = tf.shape(x)[2]  # Feature dimension
        
        pos = tf.cast(tf.range(seq_len)[:, tf.newaxis], tf.float32)
        i = tf.cast(tf.range(d_model)[tf.newaxis, :], tf.float32)
        
        angle_rates = 1 / tf.pow(10000.0, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
        angle_rads = pos * angle_rates
        
        sines = tf.sin(angle_rads[:, 0::2])
        cosines = tf.cos(angle_rads[:, 1::2])
        
        pos_encoding = tf.concat([sines, cosines], axis=-1)
        pos_encoding = pos_encoding[tf.newaxis, ...]  # Add batch dimension
        return x + pos_encoding  # Add positional encoding

    def multi_quantile_loss(self):
        def loss(y_true, y_pred):
            total_loss = 0.0
            for i, q in enumerate(self.quantiles):
                e = y_true - y_pred[:, i]
                total_loss += tf.reduce_mean(tf.maximum(q * e, (q - 1) * e))
            return total_loss / len(self.quantiles)
        return loss

    def mae_loss(self, y_true, y_pred):
        return tf.reduce_mean(tf.abs(y_true - y_pred))  # MAE Loss for each quantile

    def gated_residual_network(self, x, units):
        dense = layers.Dense(units)(x)
        gate = layers.Dense(units, activation='sigmoid')(x)
        gated = layers.Multiply()([dense, gate])
        skip = layers.Dense(units)(x)
        return layers.Add()([gated, skip])

    def load_and_preprocess_data(self):
        try:
            df = pd.read_excel(self.file_path)
            required_cols = ['Tanggal', 'Harga Penutupan', 'Jumlah Pembelian Asing', 'Jumlah Penjualan Asing']
            for col in required_cols:
                if col not in df.columns:
                    raise ValueError(f"Kolom {col} tidak ditemukan")

            if np.issubdtype(df['Tanggal'].dtype, np.number):
                df['Tanggal'] = pd.to_datetime(df['Tanggal'], unit='D', origin='1899-12-30')
            else:
                df['Tanggal'] = pd.to_datetime(df['Tanggal'], errors='coerce')
            df = df[df['Tanggal'].notna()].sort_values('Tanggal')

            ihsg_df = pd.read_excel(self.ihsg_path)
            ihsg_df['Tanggal'] = pd.to_datetime(ihsg_df['Tanggal'], errors='coerce')
            ihsg_df = ihsg_df[['Tanggal', 'IHSG']]

            df = df.merge(ihsg_df, on='Tanggal', how='left')
            df['IHSG'].fillna(method='ffill', inplace=True)
            df['IHSG'].fillna(0, inplace=True)

            df['Hari'] = df['Tanggal'].dt.day
            df['Bulan'] = df['Tanggal'].dt.month
            df['Tahun'] = df['Tanggal'].dt.year

            df['Pembelian_10_Hari'] = df['Jumlah Pembelian Asing'].rolling(window=21, min_periods=1).sum()
            df['Momentum_Harga_10'] = df['Harga Penutupan'] - df['Harga Penutupan'].shift(21)
            df['Momentum_Harga_5'] = df['Harga Penutupan'] - df['Harga Penutupan'].shift(10)
            df['Penjualan_10_Hari'] = df['Jumlah Penjualan Asing'].rolling(window=21, min_periods=1).sum()
            df['Momentum_Penjualan_10'] = df['Jumlah Penjualan Asing'] - df['Jumlah Penjualan Asing'].shift(21)
            df.fillna(0, inplace=True)

            self.original_df = df.copy()

            numeric_cols = ['Harga Penutupan', 'Jumlah Pembelian Asing', 'Jumlah Penjualan Asing',
                            'Pembelian_10_Hari', 'Momentum_Harga_10', 'Momentum_Harga_5',
                            'Penjualan_10_Hari', 'Momentum_Penjualan_10', 'IHSG']
            df[numeric_cols] = self.scaler.fit_transform(df[numeric_cols])

            return df
        except Exception as e:
            print(f"Error preprocessing: {str(e)}")
            return None

    def create_tft_dataset(self, df):
        features = ['Harga Penutupan', 'Jumlah Pembelian Asing', 'Jumlah Penjualan Asing',
                    'Hari', 'Bulan', 'Tahun',
                    'Pembelian_10_Hari', 'Momentum_Harga_10', 'Momentum_Harga_5',
                    'Penjualan_10_Hari', 'Momentum_Penjualan_10', 'IHSG']
        data = df[features].values
        X, y = [], []
        for i in range(len(data) - self.sequence_length):
            X.append(data[i:i + self.sequence_length])
            y.append(data[i + self.sequence_length, 0])
        return np.array(X), np.array(y)

    def build_tft_model(self):
        input_seq = layers.Input(shape=(self.sequence_length, 12))

        # Positional Encoding
        x = self.add_positional_encoding(input_seq)

        # Variable Selection Layer (VSN) untuk input features
        x = VariableSelectionLayer(input_dim=12)(x)
    
        # GRN pertama
        x = self.gated_residual_network(x, 64)
    
        # Attention
        attention = layers.MultiHeadAttention(num_heads=4, key_dim=16)(x, x, attention_mask=self._causal_mask())
        attention = layers.Dropout(0.1)(attention)
        x = layers.LayerNormalization()(layers.Add()([x, attention]))
    
        # GRN setelah Attention
        x = self.gated_residual_network(x, 64)
    
        # Tingkatkan kapasitas model
        lstm = layers.LSTM(128, return_sequences=True)(x)
    
        # GRN setelah LSTM
        lstm = self.gated_residual_network(lstm, 64)
    
        # Global Pooling dan Output
        pooled = layers.GlobalAveragePooling1D()(lstm)
        out = layers.Dense(32, activation='relu')(pooled)
        out = layers.Dense(3)(out)  # Untuk 3 quantile
    
        return Model(inputs=input_seq, outputs=out)

    def _causal_mask(self):
        i = tf.range(self.sequence_length)[:, None]
        j = tf.range(self.sequence_length)
        mask = tf.cast(i >= j, dtype=tf.int32)
        return tf.cast(mask[None, None, :, :], dtype=tf.float32)

    def train_model(self, X, y):
        self.model = self.build_tft_model()
        self.model.compile(optimizer='adam',
                           loss=self.multi_quantile_loss(),
                           metrics=['mae', 'mse', 'mape'])
        callbacks = [
            EarlyStopping(patience=20, restore_best_weights=True),
            CosineAnnealingScheduler(max_lr=1e-3, min_lr=1e-5, epochs=200, verbose=1)
        ]
        self.model.fit(X, y, validation_split=0.2, epochs=200, batch_size=16, callbacks=callbacks, verbose=1)

    def predict_next_day(self, last_sequence):
        last_sequence = last_sequence.reshape(1, self.sequence_length, -1)
        predicted_scaled = self.model.predict(last_sequence, verbose=0)

        q10_scaled, q50_scaled, q90_scaled = predicted_scaled[0]
        dummy = np.zeros((3, 12))
        dummy[0, 0] = q10_scaled
        dummy[1, 0] = q50_scaled
        dummy[2, 0] = q90_scaled

        inverse_result = self.scaler.inverse_transform(dummy[:, :9])
        return inverse_result[:, 0]  # [q10, q50, q90]

    def visualize_results(self, df, predicted_prices):
        plt.figure(figsize=(15, 6))

        if hasattr(self, 'original_df'):
            tanggal = self.original_df['Tanggal'].values
            harga_asli = self.original_df['Harga Penutupan'].values
        else:
            tanggal = df['Tanggal'].values
            dummy = np.zeros((len(df), 12))
            dummy[:, 0] = df['Harga Penutupan'].values
            harga_asli = self.scaler.inverse_transform(dummy[:, :9])[:, 0]

        plt.plot(tanggal, harga_asli, label='Actual')
        next_day = pd.to_datetime(tanggal[-1]) + timedelta(days=1)
        plt.plot(next_day, predicted_prices[1], 'ro', label='Forecast (q=0.5)')
        plt.fill_between(tanggal, predicted_prices[0], predicted_prices[2], color='orange', alpha=0.3, label='Prediction Range (0.1 - 0.9)')

        plt.title('STOCKS FORECASTING with IHSG')
        plt.xlabel('DATE')
        plt.ylabel('CLOSING PRICE')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()


def main():
    print("\n=== TFT STOCK FORECASTING (v5-Improved MQ) with IHSG ===")
    forecaster = TFTStockForecaster()
    df = forecaster.load_and_preprocess_data()
    if df is None:
        return

    X, y = forecaster.create_tft_dataset(df)
    print("\nTraining model...")
    forecaster.train_model(X, y)

    features = ['Harga Penutupan', 'Jumlah Pembelian Asing', 'Jumlah Penjualan Asing',
                'Hari', 'Bulan', 'Tahun',
                'Pembelian_10_Hari', 'Momentum_Harga_10', 'Momentum_Harga_5',
                'Penjualan_10_Hari', 'Momentum_Penjualan_10', 'IHSG']
    last_seq = df[features].values[-forecaster.sequence_length:]
    forecast = forecaster.predict_next_day(last_seq)

    print(f"\nPrediksi harga besok:")
    for q, val in zip([0.1, 0.5, 0.9], forecast):
        print(f"Quantile {q:.1f}: {val:.2f}")

    forecaster.visualize_results(df, forecast)


if __name__ == "__main__":
    main()
