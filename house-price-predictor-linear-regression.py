from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import pandas as pd
from sklearn.linear_model import LinearRegression

# دادهات مثل قبل
data = pd.DataFrame({
    'metraj': [50, 80, 100, 120, 150, 70, 90, 110],
    'sal_sakht': [1390, 1395, 1400, 1385, 1398, 1392, 1396, 1399],
    'tabaghe': [2, 3, 4, 2, 5, 3, 3, 4],
    'otagh': [1, 2, 3, 3, 4, 2, 2, 3],
    'asansor': [0, 1, 1, 0, 1, 0, 1, 1],
    'parking': [1, 1, 1, 0, 1, 0, 1, 1],
    'anbari': [0, 1, 1, 1, 1, 0, 0, 1],
    'no_sazi': [1, 0, 0, 2, 1, 1, 0, 0],
    'price': [500, 800, 1000, 1200, 1500, 650, 900, 1100]
})  # همون داده خودت

X = data[['metraj', 'sal_sakht', 'tabaghe', 'otagh', 'asansor', 'parking', 'anbari', 'no_sazi']]
y = data['price']

# ۱. تقسیم به آموزش و تست (۲۰٪ تست)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ۲. استانداردسازی (خیلی مهم!)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ۳. مدل جدید
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# ۴. ارزیابی روی دادههای تست
y_pred = model.predict(X_test_scaled)
print(f"خطای مطلق متوسط: {mean_absolute_error(y_test, y_pred):.0f} تومان")
print(f"ضریب تعیین R²: {r2_score(y_test, y_pred):.2f}")