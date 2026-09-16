import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# 1. Đọc dữ liệu từ file CSV cục bộ
print("Đang tải dữ liệu từ file data.csv...")
data = pd.read_csv('data.csv')

# Khai báo các cột đặc trưng (X) và cột mục tiêu (y)
X = data[['DienTich', 'SoPhongNgu', 'TuoiNha']]
y = data['GiaNha']

# 2. Chia tập dữ liệu thành Train và Test (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Khởi tạo và huấn luyện mô hình Hồi quy tuyến tính
print("Đang huấn luyện mô hình Hồi quy tuyến tính...")
model = LinearRegression()
model.fit(X_train, y_train)

# 4. Dự đoán trên tập Test và đánh giá
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

print("Hoàn tất!")
print(f"Sai số toàn phương trung bình (MSE): {mse:.2f}")
