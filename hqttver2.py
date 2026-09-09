import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

print("--- CHƯƠNG TRÌNH DỰ BÁO GIÁ NHÀ (VERSION 2) ---")
print("Đang tải dữ liệu...")

# Lưu ý: Sửa lại tên file 'dbgianha.csv' hoặc 'data.csv' cho khớp với tên file thực tế của bạn
data = pd.read_csv('dbgianha.csv') 

# Chọn các cột đặc trưng (X) và cột mục tiêu (y)
X = data[['LotArea', 'YearBuilt']] 
y = data['SalePrice'] 

# Chia tập dữ liệu thành Train và Test (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Khởi tạo và huấn luyện mô hình Hồi quy tuyến tính
print("Đang huấn luyện mô hình Hồi quy tuyến tính...")
model = LinearRegression()
model.fit(X_train, y_train)

# Dự đoán trên tập Test và đánh giá
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)  # ĐIỂM MỚI Ở VERSION 2: Tính thêm độ phù hợp của mô hình

print("Hoàn tất!")
print(f"Sai số toàn phương trung bình (MSE): {mse:.2f}")
print(f"Độ chính xác của mô hình (R^2 Score): {r2:.4f}")