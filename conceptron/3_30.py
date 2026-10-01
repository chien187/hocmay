import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler

# 1. Tải tập dữ liệu Breast Cancer
data = load_breast_cancer()
X = data.data
y = data.target # 0: ác tính (malignant), 1: lành tính (benign)

# 2. Chia tập huấn luyện và tập kiểm tra (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Chuẩn hóa dữ liệu (Perceptron rất nhạy cảm với scale của dữ liệu)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Khởi tạo và huấn luyện mô hình Perceptron
clf = Perceptron(max_iter=1000, eta0=0.1, random_state=42)
clf.fit(X_train_scaled, y_train)

# 4. Dự báo trên tập test
y_pred = clf.predict(X_test_scaled)

# 5. Đánh giá kết quả
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("--- KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH PERCEPTRON (BREAST CANCER DATASET) ---")
print(f"Accuracy  (Độ chính xác)     : {acc:.4f}")
print(f"Precision (Độ chuẩn xác)     : {prec:.4f}")
print(f"Recall    (Độ bao phủ)       : {rec:.4f}")
print(f"F1-score  (Trung bình điều hòa): {f1:.4f}")