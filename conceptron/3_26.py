def f(x):
    return x**2 - 4*x + 5

def df(x):
    return 2*x - 4

# Khởi tạo
x = 5
learning_rate = 0.2
steps = 4

print(f"Bước 0: x = {x:.4f}, f(x) = {f(x):.4f}")
for i in range(1, steps + 1):
    grad = df(x)
    x = x - learning_rate * grad
    print(f"Bước {i}: x = {x:.4f}, f(x) = {f(x):.4f}")