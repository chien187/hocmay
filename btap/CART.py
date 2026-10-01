import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import matplotlib.pyplot as plt

# 1. Khởi tạo tập dữ liệu
data = {
    'age': ['<=30', '<=30', '31...40', '>40', '>40', '>40', '31...40', '<=30', '<=30', '>40', '<=30', '31...40', '31...40', '>40'],
    'income': ['high', 'high', 'high', 'medium', 'low', 'low', 'low', 'medium', 'low', 'medium', 'medium', 'medium', 'high', 'medium'],
    'student': ['no', 'no', 'no', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'no'],
    'credit_rating': ['fair', 'excellent', 'fair', 'fair', 'fair', 'excellent', 'excellent', 'fair', 'fair', 'fair', 'excellent', 'excellent', 'fair', 'excellent'],
    'buys_computer': ['no', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'no', 'yes', 'yes', 'yes', 'yes', 'yes', 'no']
}
df = pd.DataFrame(data)

# 2. Tiền xử lý dữ liệu: Chuyển đổi chuỗi sang số
label_encoders = {}
for column in df.columns:
    le = LabelEncoder()
    df[column + '_n'] = le.fit_transform(df[column])
    label_encoders[column] = le

X = df[['age_n', 'income_n', 'student_n', 'credit_rating_n']]
y = df['buys_computer_n']
feature_names = ['age', 'income', 'student', 'credit_rating']
class_names = label_encoders['buys_computer'].classes_

# 3. Huấn luyện mô hình CART với criterion='gini'
clf_cart = DecisionTreeClassifier(criterion='gini', random_state=42)
clf_cart.fit(X, y)

# 4. Trực quan hóa cây quyết định
plt.figure(figsize=(10, 6))
tree.plot_tree(clf_cart, 
               feature_names=feature_names,  
               class_names=class_names,
               filled=True, 
               rounded=True)
plt.title("Decision Tree - CART (Gini Index)")
plt.show()