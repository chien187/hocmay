import numpy as np

w = np.array([1, 2, -10])
x = np.array([3, 4, 1])
y_true = -1

# 1. Tính w^T * x
w_tx = np.dot(w, x)
print(f"1. w^T * x = {w_tx}")

# 2. Xác định nhãn dự đoán (sử dụng hàm sign)
y_pred = 1 if w_tx >= 0 else -1
print(f"2. Nhãn dự đoán: {y_pred}")

# 3. Kiểm tra phân lớp
if y_pred != y_true:
    print("3. Điểm dữ liệu bị phân lớp SAI.")
else:
    print("3. Điểm dữ liệu được phân lớp ĐÚNG.")