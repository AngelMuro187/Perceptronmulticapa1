import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, models

# === Carga de datos ===
df = pd.read_csv('Extended_Employee_Performance_and_Productivity_Data.csv')

print("Información del DataFrame:")
print(df.info())

# Filtrar las columnas numéricas y excluir Employee_ID
numeric_columns = df.select_dtypes(include=['number']).drop('Employee_ID', axis=1)

# Listado de columnas numéricas
cols = list(numeric_columns.columns)

# Histograma de cada variable numérica
fig, axes = plt.subplots(1, len(cols), figsize=(5 * len(cols), 4))

for i, col in enumerate(cols):
    axes[i].hist(df[col], bins=20, color='skyblue', edgecolor='black')
    axes[i].set_title(col)
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Frecuencia')

plt.tight_layout()
plt.show()

# === Actividad 1: Preparación de datos ===

# Tomamos X como todas las columnas numéricas excepto la variable objetivo
X = numeric_columns.drop('Employee_Satisfaction_Score', axis=1)

# Variable objetivo
y = numeric_columns['Employee_Satisfaction_Score']

# Convertimos la satisfacción a 5 categorías: 0,1,2,3,4
y = y.apply(lambda x: round(x) - 1)

# Estandarización
scaler = StandardScaler()
X_standar = scaler.fit_transform(X)

# División en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X_standar, y, test_size=0.33, random_state=42
)

# One-hot encoding de la variable objetivo
y_onehot_train = tf.keras.utils.to_categorical(y_train, 5)
y_onehot_test = tf.keras.utils.to_categorical(y_test, 5)

input_dim = X_train.shape[1]

# === Actividad 2: Implementación de 3 arquitecturas MLP ===

# Modelo 1 – Arquitectura sencilla
model_mlp_1 = models.Sequential([
    layers.Input(shape=(input_dim,)),
    layers.Dense(32, activation='relu'),
    layers.Dense(5, activation='softmax')
], name="mlp_model_1")

# Modelo 2 – Arquitectura intermedia con dropout
model_mlp_2 = models.Sequential([
    layers.Input(shape=(input_dim,)),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(32, activation='relu'),
    layers.Dense(5, activation='softmax')
], name="mlp_model_2")

# Modelo 3 – Arquitectura más profunda con batch normalization y dropout
model_mlp_3 = models.Sequential([
    layers.Input(shape=(input_dim,)),
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.4),
    layers.Dense(32, activation='relu'),
    layers.Dense(5, activation='softmax')
], name="mlp_model_3")

print("\nResumen de modelos:")
model_mlp_1.summary()
model_mlp_2.summary()
model_mlp_3.summary()

# === Actividad 3: Compilación y entrenamiento de los modelos ===

def compile_and_train(model, X_train, y_train, X_test, y_test, epochs=30, batch_size=32):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )
    return history

print("\nEntrenando modelo 1...")
history1 = compile_and_train(model_mlp_1, X_train, y_onehot_train, X_test, y_onehot_test)

print("\nEntrenando modelo 2...")
history2 = compile_and_train(model_mlp_2, X_train, y_onehot_train, X_test, y_onehot_test)

print("\nEntrenando modelo 3...")
history3 = compile_and_train(model_mlp_3, X_train, y_onehot_train, X_test, y_onehot_test)

# Evaluación final en el conjunto de prueba
print("\nEvaluación en el conjunto de prueba:")
for model in [model_mlp_1, model_mlp_2, model_mlp_3]:
    loss, acc = model.evaluate(X_test, y_onehot_test, verbose=0)
    print(f"{model.name}: loss={loss:.4f}, accuracy={acc:.4f}")
