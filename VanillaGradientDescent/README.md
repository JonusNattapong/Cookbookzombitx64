# Vanilla Gradient Descent

โปรเจกต์นี้สำหรับการศึกษาและทดลองใช้งานอัลกอริทึม Vanilla Gradient Descent สำหรับปัญหาการถดถอยเชิงเส้น (Linear Regression)

## หลักการ

Vanilla Gradient Descent เป็นอัลกอริทึมการเรียนรู้ของเครื่องที่ใช้ในการหาค่าต่ำสุดของฟังก์ชัน (ในที่นี้คือฟังก์ชันความสูญเสีย) โดยการคำนวณ gradient ของฟังก์ชันความสูญเสียจากข้อมูลทั้งชุด (full batch) แล้วอัปเดตพารามิเตอร์ทีละขั้น

โดยมีขั้นตอนดังนี้:
1. กำหนดค่าเริ่มต้นให้กับพารามิเตอร์ (น้ำหนักและค่าคงที่)
2. คำนวณการทำนายโดยใช้พารามิเตอร์ปัจจุบัน
3. คำนวณค่าความสูญเสีย (loss) จากการทำนายและค่าจริง
4. คำนวณ gradient ของฟังก์ชันความสูญเสียเทียบกับพารามิเตอร์แต่ละตัว
5. อัปเดตพารามิเตอร์โดยใช้ gradient และอัตราการเรียนรู้ (learning rate)
6. ทำซ้ำขั้นตอนที่ 2-5 จนกว่าจะถึงเงื่อนไขการหยุด

## ไฟล์ในโปรเจกต์

โปรเจกต์นี้ประกอบด้วยไฟล์ต่าง ๆ ดังนี้:

- `dataset_generator.py`: สคริปต์สำหรับสร้างชุดข้อมูลสังเคราะห์สำหรับปัญหาการถดถอยเชิงเส้น
- `vanilla_gradient_descent.py`: คลาสที่ใช้ในการทำ Vanilla Gradient Descent
- `run_demo.py`: สคริปต์สำหรับแสดงการทำงานของอัลกอริทึม

## การใช้งาน

1. สร้างและแสดงภาพชุดข้อมูล:
```python
from dataset_generator import generate_linear_dataset, visualize_dataset

# สร้างชุดข้อมูล
X, y = generate_linear_dataset(n_samples=100, n_features=1, noise=15.0)

# แสดงภาพชุดข้อมูล
visualize_dataset(X, y)
```

2. ฝึกโมเดลด้วย Vanilla Gradient Descent:
```python
from vanilla_gradient_descent import VanillaGradientDescent

# สร้างและฝึกโมเดล
model = VanillaGradientDescent(learning_rate=0.1, max_iterations=1000)
model.fit(X, y)

# ทำนายและแสดงผลลัพธ์
y_pred = model.predict(X)
mse = model.compute_loss(y, y_pred)
print(f"ค่าน้ำหนัก: {model.weights}")
print(f"ค่า Bias: {model.bias}")
print(f"ค่าความสูญเสีย (MSE): {mse:.6f}")
```

3. แสดงกราฟผลลัพธ์:
```python
# แสดงกราฟประวัติค่าความสูญเสีย
model.plot_loss_history()

# แสดงกราฟเส้นถดถอย
model.plot_regression_line(X, y)
```

4. รันการสาธิตทั้งหมด:
```bash
python run_demo.py
```

## ข้อกำหนดเบื้องต้น

- Python 3.6 ขึ้นไป
- NumPy
- Matplotlib
- scikit-learn

## การติดตั้ง

```bash
pip install numpy matplotlib scikit-learn
``` 