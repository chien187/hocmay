import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline

print("Đang tải dữ liệu...")
data = pd.read_csv('data.csv')

X = data[['DienTich', 'TuoiNha']]
y = data['GiaNha']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\n--- TRƯỜNG HỢP 1: XẢY RA OVERFITTING ---")
# Dùng đa thức bậc 8 (degree=8) để ép mô hình bám sát từng điểm dữ liệu Train
# StandardScaler dùng để chuẩn hóa dữ liệu, tránh lỗi số quá lớn khi tính mũ 8
overfit_model = make_pipeline(StandardScaler(), PolynomialFeatures(degree=8), LinearRegression())
overfit_model.fit(X_train, y_train)

train_r2_overfit = r2_score(y_train, overfit_model.predict(X_train))
test_r2_overfit = r2_score(y_test, overfit_model.predict(X_test))

print(f"Điểm R^2 trên tập Train: {train_r2_overfit:.4f} (Điểm cao -> Học thuộc rất tốt)")
print(f"Điểm R^2 trên tập Test : {test_r2_overfit:.4f} (Điểm âm -> Dự đoán thực tế cực tệ, sụp đổ hoàn toàn)")



print("\n--- TRƯỜNG HỢP 2: KHẮC PHỤC BẰNG RIDGE REGRESSION ---")
# Vẫn dùng đa thức bậc 8, nhưng thay LinearRegression bằng Ridge (kỹ thuật L2 Regularization)
# Tham số alpha=100.0 sẽ phạt nặng các trọng số ảo tưởng, ép mô hình bớt ngoằn ngoèo lại
regularized_model = make_pipeline(StandardScaler(), PolynomialFeatures(degree=8), Ridge(alpha=100.0))
regularized_model.fit(X_train, y_train)

train_r2_reg = r2_score(y_train, regularized_model.predict(X_train))
test_r2_reg = r2_score(y_test, regularized_model.predict(X_test))

print(f"Điểm R^2 trên tập Train: {train_r2_reg:.4f}")
print(f"Điểm R^2 trên tập Test : {test_r2_reg:.4f} (Điểm đã dương trở lại, mô hình ổn định và thực tế hơn)")