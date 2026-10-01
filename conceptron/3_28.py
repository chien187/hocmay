import numpy as np

w = np.array([-2, 1, 0])
x = np.array([2, 3, 1])
y_true = 1

w_tx = np.dot(w, x)
y_pred = 1 if w_tx >= 0 else -1

# 1. Kiểm tra
print(f"w^T * x ban đầu = {w_tx}")
print(f"Nhãn dự đoán ban đầu = {y_pred}")
if y_pred != y_true:
    print("-> Mẫu bị phân lớp SAI, tiến hành cập nhật.")
    
    # 2. Thực hiện cập nhật w = w + y*x (giả định learning rate = 1)
    w_new = w + y_true * x
    print(f"-> Trọng số w mới sau cập nhật: {w_new}")
    
    # 3. Tính lại w^T * x
    w_tx_new = np.dot(w_new, x)
    print(f"-> w^T * x sau cập nhật: {w_tx_new}")
    print(f"-> Nhãn dự đoán mới: {1 if w_tx_new >= 0 else -1} (Khớp với y_true = {y_true})")