# Stochastic Gradient Descent (SGD)

โปรเจกต์นี้สำหรับการศึกษาและทดลองใช้งานอัลกอริทึม Stochastic Gradient Descent (SGD) สำหรับปัญหาการถดถอยเชิงเส้น (Linear Regression)

## หลักการ

Stochastic Gradient Descent (SGD) เป็นอัลกอริทึมสำหรับการเรียนรู้ของเครื่องที่มีแนวคิดดังนี้:
- อัปเดตน้ำหนักโดยใช้ gradient จากข้อมูลตัวอย่างเดียว (หรือ batch เล็กๆ) แทนทั้งชุด
- สุ่มเลือกข้อมูล 1 ตัวอย่างหรือ batch
- คำนวณ gradient และอัปเดตน้ำหนักทันที

โดยมีขั้นตอนดังนี้:
1. กำหนดค่าเริ่มต้นให้กับพารามิเตอร์ (น้ำหนักและค่าคงที่)
2. สำหรับแต่ละรอบการเรียนรู้ (epoch):
   - สุ่มลำดับของข้อมูล
   - แบ่งข้อมูลเป็น batch ย่อยๆ
   - สำหรับแต่ละ batch:
     - คำนวณการทำนายโดยใช้พารามิเตอร์ปัจจุบัน
     - คำนวณ gradient ของฟังก์ชันความสูญเสียเทียบกับพารามิเตอร์แต่ละตัว
     - อัปเดตพารามิเตอร์โดยใช้ gradient และอัตราการเรียนรู้ (learning rate)
3. ทำซ้ำจนกว่าจะถึงเงื่อนไขการหยุด

## จุดเด่น
- เร็วกว่า Vanilla Gradient Descent เนื่องจากไม่ต้องใช้ข้อมูลทั้งหมดในการคำนวณ gradient
- เหมาะกับชุดข้อมูลขนาดใหญ่
- อาจช่วยให้หลีกเลี่ยงการติดอยู่ใน local minimum

## ข้อจำกัด
- gradient มีความผันผวนสูง ทำให้การเรียนรู้ไม่เสถียร
- อาจต้องใช้เวลานานกว่าในการลู่เข้า
- อาจต้องปรับ learning rate ให้เหมาะสม

## ไฟล์ในโปรเจกต์

โปรเจกต์นี้ประกอบด้วยไฟล์ต่าง ๆ ดังนี้:

- `stochastic_gradient_descent.py`: คลาสที่ใช้ในการทำ Stochastic Gradient Descent
- `run_demo.py`: สคริปต์สำหรับแสดงการทำงานของอัลกอริทึมและเปรียบเทียบขนาด batch ต่างๆ

## การใช้งาน

1. ฝึกโมเดลด้วย Stochastic Gradient Descent:
```python
from stochastic_gradient_descent import StochasticGradientDescent

# สร้างและฝึกโมเดล
model = StochasticGradientDescent(learning_rate=0.1, epochs=50, batch_size=10)
model.fit(X, y)

# ทำนายและแสดงผลลัพธ์
y_pred = model.predict(X)
mse = model.compute_loss(y, y_pred)
print(f"ค่าน้ำหนัก: {model.weights}")
print(f"ค่า Bias: {model.bias}")
print(f"ค่าความสูญเสีย (MSE): {mse:.6f}")
```

2. แสดงกราฟผลลัพธ์:
```python
# แสดงกราฟประวัติค่าความสูญเสีย
model.plot_loss_history()

# แสดงกราฟเส้นถดถอย
model.plot_regression_line(X, y)
```

3. รันการสาธิตและเปรียบเทียบขนาด batch:
```bash
python run_demo.py
```

## เปรียบเทียบกับ Vanilla Gradient Descent

| คุณลักษณะ                  | Vanilla Gradient Descent    | Stochastic Gradient Descent     |
|----------------------------|----------------------------|---------------------------------|
| ขนาดข้อมูลที่ใช้คำนวณ gradient | ข้อมูลทั้งชุด                | 1 ตัวอย่างหรือ batch เล็กๆ        |
| ความเร็วในการเรียนรู้          | ช้ากว่า                     | เร็วกว่า                         |
| ความเสถียร                  | มีความเสถียรสูง               | มีความผันผวน                     |
| หน่วยความจำที่ใช้             | มากกว่า                     | น้อยกว่า                        |
| เหมาะกับข้อมูลขนาด            | เล็ก                       | ใหญ่                           |

## ข้อกำหนดเบื้องต้น

- Python 3.6 ขึ้นไป
- NumPy
- Matplotlib
- scikit-learn (สำหรับการสร้างชุดข้อมูล) 