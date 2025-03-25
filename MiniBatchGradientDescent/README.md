# Mini-Batch Gradient Descent

โปรเจกต์นี้สำหรับการศึกษาและทดลองใช้งานอัลกอริทึม Mini-Batch Gradient Descent สำหรับปัญหาการถดถอยเชิงเส้น (Linear Regression)

## หลักการ

Mini-Batch Gradient Descent เป็นอัลกอริทึมสำหรับการเรียนรู้ของเครื่องที่มีแนวคิดดังนี้:
- ผสมข้อดีของ Vanilla Gradient Descent และ Stochastic Gradient Descent
- ใช้ batch ขนาดเล็ก (เช่น 32, 64 ตัวอย่าง) ในการคำนวณ gradient
- แบ่งข้อมูลเป็น batch และคำนวณ gradient และอัปเดตน้ำหนักสำหรับแต่ละ batch

โดยมีขั้นตอนดังนี้:
1. กำหนดค่าเริ่มต้นให้กับพารามิเตอร์ (น้ำหนักและค่าคงที่)
2. แบ่งข้อมูลทั้งหมดเป็น batch ขนาดเล็กๆ (เช่น 32, 64 ตัวอย่าง)
3. สำหรับแต่ละรอบการเรียนรู้ (epoch):
   - สุ่มลำดับของ batch
   - สำหรับแต่ละ batch:
     - คำนวณการทำนายโดยใช้พารามิเตอร์ปัจจุบัน
     - คำนวณ gradient ของฟังก์ชันความสูญเสียเทียบกับพารามิเตอร์แต่ละตัว
     - อัปเดตพารามิเตอร์โดยใช้ gradient และอัตราการเรียนรู้ (learning rate)
4. ทำซ้ำจนกว่าจะถึงเงื่อนไขการหยุด

## จุดเด่น
- สมดุลระหว่างความเร็วและความเสถียร
- เป็นวิธีที่ใช้กันมากที่สุดในการฝึกโมเดล deep learning
- ทำงานได้ดีกับชุดข้อมูลขนาดใหญ่
- มีความเสถียรมากกว่า SGD แต่ยังคงมีประสิทธิภาพในการหลีกเลี่ยง local minima

## ข้อจำกัด
- ต้องเลือกขนาด batch size ที่เหมาะสม
- ต้องการหน่วยความจำมากกว่า SGD แต่น้อยกว่า Vanilla Gradient Descent
- อาจต้องปรับพารามิเตอร์หลายตัวเพื่อให้ได้ผลลัพธ์ที่ดี

## ไฟล์ในโปรเจกต์

โปรเจกต์นี้ประกอบด้วยไฟล์ต่าง ๆ ดังนี้:

- `mini_batch_gradient_descent.py`: คลาสที่ใช้ในการทำ Mini-Batch Gradient Descent ทั้งแบบใช้ PyTorch และแบบ NumPy ปกติ
- `run_demo.py`: สคริปต์สำหรับแสดงการทำงานของอัลกอริทึมและเปรียบเทียบขนาด batch และวิธีการหาค่าเหมาะสมต่างๆ

## การใช้งาน

1. ฝึกโมเดลด้วย Mini-Batch Gradient Descent (ด้วย PyTorch):
```python
from mini_batch_gradient_descent import MiniBatchGradientDescent

# สร้างและฝึกโมเดล
model = MiniBatchGradientDescent(learning_rate=0.1, epochs=50, batch_size=32, use_pytorch=True)
model.fit(X, y)

# ทำนายและแสดงผลลัพธ์
y_pred = model.predict(X)
mse = model.compute_loss(y, y_pred)
print(f"ค่าน้ำหนัก: {model.weights}")
print(f"ค่า Bias: {model.bias}")
print(f"ค่าความสูญเสีย (MSE): {mse:.6f}")
```

2. ฝึกโมเดลด้วย Mini-Batch Gradient Descent (ไม่ใช้ PyTorch):
```python
from mini_batch_gradient_descent import MiniBatchGradientDescent

# สร้างและฝึกโมเดล
model = MiniBatchGradientDescent(learning_rate=0.1, epochs=50, batch_size=32, use_pytorch=False)
model.fit(X, y)
```

3. เปรียบเทียบขนาด batch size ต่างๆ:
```python
from run_demo import compare_batch_sizes

# เปรียบเทียบผลลัพธ์ของขนาด batch size ต่างๆ
results = compare_batch_sizes(n_samples=100, n_features=1, noise=15.0, learning_rate=0.1, epochs=50)
```

4. เปรียบเทียบวิธีการหาค่าเหมาะสมต่างๆ:
```python
from run_demo import compare_optimization_methods

# เปรียบเทียบผลลัพธ์ของวิธีการหาค่าเหมาะสมต่างๆ
results = compare_optimization_methods(n_samples=100, n_features=1, noise=15.0, learning_rate=0.1, epochs=50)
```

5. รันการสาธิตทั้งหมด:
```bash
python run_demo.py
```

## เปรียบเทียบกับ Vanilla GD และ SGD

| คุณลักษณะ                  | Vanilla GD               | SGD                       | Mini-Batch GD             |
|----------------------------|--------------------------|-----------------------------|---------------------------|
| ขนาดข้อมูลที่ใช้คำนวณ gradient | ข้อมูลทั้งชุด              | 1 ตัวอย่าง                  | batch ขนาดเล็ก (เช่น 32, 64) |
| ความเร็วในการเรียนรู้         | ช้า                      | เร็วที่สุด                  | ปานกลาง                   |
| ความเสถียร                 | มีความเสถียรสูง            | มีความผันผวนสูง              | มีความเสถียรปานกลาง        |
| หน่วยความจำที่ใช้            | มากที่สุด                 | น้อยที่สุด                  | ปานกลาง                   |
| เหมาะกับข้อมูลขนาด           | เล็ก                     | ใหญ่มาก                    | ใหญ่                      |
| การหลีกเลี่ยง local minima   | ไม่ดี                    | ดีที่สุด                    | ดี                        |
| การใช้งานในปัจจุบัน          | น้อย                     | ปานกลาง                    | มากที่สุด                  |

## ข้อกำหนดเบื้องต้น

- Python 3.6 ขึ้นไป
- NumPy
- Matplotlib
- PyTorch (ตัวเลือก - ถ้าต้องการใช้ DataLoader)
- scikit-learn (สำหรับการสร้างชุดข้อมูล) 