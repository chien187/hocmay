import numpy as np

class Perceptron:
    def __init__(self, learning_rate=0.1, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        # Số lượng mẫu (n_samples) và số đặc trưng (n_features)
        n_samples, n_features = X.shape
        
        # Khởi tạo trọng số và bias bằng 0
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Đảm bảo nhãn y là 1 và -1
        y_ = np.where(y <= 0, -1, 1)

        # Huấn luyện
        for _ in range(self.n_iterations):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = np.sign(linear_output)
                
                # Cập nhật nếu dự đoán sai
                if y_[idx] * linear_output <= 0:
                    update = self.learning_rate * y_[idx]
                    self.weights += update * x_i
                    self.bias += update

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        return np.where(linear_output >= 0, 1, -1)

# Đoạn code test nhỏ để kiểm tra class
if __name__ == "__main__":
    # Dữ liệu test logic AND
    X_train = np.array([[0,0], [0,1], [1,0], [1,1]])
    y_train = np.array([-1, -1, -1, 1])
    
    model = Perceptron(learning_rate=0.1, n_iterations=10)
    model.fit(X_train, y_train)
    predictions = model.predict(X_train)
    print(f"Dự đoán mô hình test: {predictions}")