from utils import db_connect
engine = db_connect()

# === 1. Librerías necesarias ===
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
import joblib

# === 2. Carga del dataset ===

# Cargar el dataset desde la URL
url = "https://raw.githubusercontent.com/4GeeksAcademy/k-means-project-tutorial/main/housing.csv"

# === 3. División train/test ===
train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

# === 4. Escalado de features para clustering ===
scaler = StandardScaler()
X_train_cluster = train_data[['Latitude', 'Longitude', 'MedInc']]
X_test_cluster = test_data[['Latitude', 'Longitude', 'MedInc']]
X_scaled_train = scaler.fit_transform(X_train_cluster)
X_scaled_test = scaler.transform(X_test_cluster)

# === 5. KMeans con k=4 ===
kmeans_4 = KMeans(n_clusters=4, random_state=42, n_init=10)
train_data['cluster'] = kmeans_4.fit_predict(X_scaled_train)
test_data['cluster'] = kmeans_4.predict(X_scaled_test)

# === 6. Visualización de clusters geográficos ===
plt.figure(figsize=(10, 6))
sns.scatterplot(data=train_data, x='Longitude', y='Latitude', hue='cluster',
                palette='tab10', alpha=0.3, s=40)
sns.scatterplot(data=test_data, x='Longitude', y='Latitude', hue='cluster',
                palette='tab10', alpha=1.0, s=10, legend=False)
plt.title("Comparación de clusters (k=4) - Train vs Test")
plt.xlabel("Longitud")
plt.ylabel("Latitud")
plt.grid(True)
plt.show()

# === 7. Clasificación supervisada con XGBoost ===
features_sup = ['HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'MedHouseVal']
X_train_sup = train_data[features_sup]
y_train_sup = train_data['cluster']
X_test_sup = test_data[features_sup]
y_test_sup = test_data['cluster']

xgb_final = XGBClassifier(
    objective='multi:softmax',
    num_class=4,
    eval_metric='mlogloss',
    use_label_encoder=False,
    random_state=42
)
xgb_final.fit(X_train_sup, y_train_sup)

# === 8. Evaluación del clasificador ===
y_pred_final = xgb_final.predict(X_test_sup)
print("=== XGBoostClassifier (con k=4) ===")
print(classification_report(y_test_sup, y_pred_final))

# === 9. Guardado de modelos ===
joblib.dump(kmeans_4, 'kmeans_model_k4.joblib')
joblib.dump(scaler, 'scaler_for_kmeans.joblib')
joblib.dump(xgb_final, 'xgb_classifier_k4.joblib')
print("Modelos guardados: kmeans_model_k4.joblib, scaler_for_kmeans.joblib, xgb_classifier_k4.joblib")
