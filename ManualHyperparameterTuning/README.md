# การปรับ Hyperparameters แบบ Manual

การปรับ hyperparameters แบบ manual เป็นวิธีการพื้นฐานในการหาค่าที่เหมาะสมที่สุดสำหรับโมเดล machine learning และ deep learning ซึ่งต้องอาศัยการทดลองและความเข้าใจเกี่ยวกับผลกระทบของ hyperparameters ต่างๆ

## แนวคิด

- ปรับ hyperparameters (เช่น learning rate, batch size, model size) ด้วยมือโดยลองค่าต่างๆ
- วัดประสิทธิภาพของแต่ละชุดค่า hyperparameters บนข้อมูล validation set
- เลือกชุดค่า hyperparameters ที่ให้ผลลัพธ์ดีที่สุด

## วิธีการ

1. **เลือก Hyperparameters ที่จะปรับ**:
   - Learning rate
   - Batch size
   - จำนวน hidden units/layers
   - Optimizer
   - Regularization strength
   - Activation functions

2. **กำหนดช่วงค่าที่จะลอง**:
   - สำหรับ learning rate: [0.001, 0.01, 0.1]
   - สำหรับ batch size: [16, 32, 64]
   - สำหรับ hidden size: [32, 64, 128]

3. **ใช้ Grid Search**:
   - ลองทุกการรวมกันของ hyperparameters
   - เทรนและประเมินผลโมเดลสำหรับแต่ละชุดค่า

4. **บันทึกผลลัพธ์**:
   - เก็บค่าความแม่นยำ (accuracy), F1 score, loss
   - เก็บเวลาที่ใช้ในการเทรน

5. **เลือกชุดค่าที่ดีที่สุด**:
   - เลือกจาก validation accuracy หรือ F1 score

## เครื่องมือที่ใช้

- PyTorch หรือ TensorFlow/Keras สำหรับการเทรนโมเดล
- Matplotlib และ Pandas สำหรับการวิเคราะห์และแสดงผลลัพธ์

## จุดเด่น

1. **เรียบง่าย**: ไม่ต้องใช้เครื่องมือหรือไลบรารีเพิ่มเติม

2. **เข้าใจผลกระทบ**: ช่วยให้เข้าใจความสัมพันธ์ระหว่าง hyperparameters และประสิทธิภาพ

3. **การควบคุม**: ผู้พัฒนามีการควบคุมเต็มที่ในการเลือกค่าที่จะลอง

4. **ความยืดหยุ่น**: สามารถปรับกระบวนการและเปลี่ยนแปลงได้ง่ายตามความต้องการ

## ข้อจำกัด

1. **ใช้เวลานาน**: ต้องลองหลายการรวมกันของ hyperparameters

2. **ต้องใช้ทรัพยากรมาก**: การเทรนหลายโมเดลต้องใช้ทรัพยากรการคำนวณมาก

3. **อาจไม่ได้ค่าที่ดีที่สุด**: การสุ่มเลือกช่วงค่าอาจไม่ครอบคลุมค่าที่ดีที่สุด

4. **ล้าสมัย**: เมื่อเทียบกับเทคนิคอัตโนมัติอย่าง Bayesian Optimization

## การใช้งานในโค้ดตัวอย่าง

โค้ดตัวอย่างนี้แสดงวิธีการปรับ hyperparameters แบบ manual โดยใช้ grid search:

1. ติดตั้ง dependencies:
```bash
pip install torch numpy matplotlib scikit-learn pandas tqdm tabulate
```

2. รันโค้ดตัวอย่าง:
```bash
python manual_hyperparameter_tuning.py
```

3. ผลลัพธ์ที่ได้จะแสดง:
   - กราฟเปรียบเทียบประสิทธิภาพของ hyperparameters ต่างๆ
   - ตารางสรุปผลลัพธ์ของทุกการทดลอง
   - ชุดค่า hyperparameters ที่ดีที่สุด และผลลัพธ์
   - ข้อสังเกตเกี่ยวกับผลกระทบของแต่ละ hyperparameter

## การวิเคราะห์ผลลัพธ์

1. **Learning Rate**:
   - ค่าที่เล็กเกินไป: การเรียนรู้ช้า, อาจติดใน local minima
   - ค่าที่ใหญ่เกินไป: อาจไม่ลู่เข้า, loss ผันผวนมาก
   - ค่าที่เหมาะสม: จะทำให้ loss ลดลงอย่างสม่ำเสมอ

2. **Batch Size**:
   - ค่าที่เล็ก: ใช้หน่วยความจำน้อย, อัพเดทบ่อย, loss ผันผวนมาก
   - ค่าที่ใหญ่: ใช้หน่วยความจำมาก, อัพเดทน้อยครั้ง, loss เรียบกว่า
   - ค่าที่เหมาะสม: สมดุลระหว่างความเร็วและความเสถียร

3. **Hidden Size**:
   - ค่าที่เล็ก: โมเดลอาจจะ underfit, เรียนรู้ได้น้อย
   - ค่าที่ใหญ่: โมเดลอาจจะ overfit, ใช้ทรัพยากรมาก
   - ค่าที่เหมาะสม: จะมีความสามารถในการทำนายที่ดีโดยไม่ overfit

4. **Optimizer**:
   - Adam: โดยทั่วไปลู่เข้าเร็ว, ปรับตัวได้ดี
   - SGD: อาจต้องการการปรับ learning rate มากกว่า
   - RMSprop: ทำงานได้ดีกับ RNN

## เทคนิคเพิ่มเติม

1. **Random Search**: สุ่มเลือก hyperparameters แทนการลองทุกการรวมกัน

2. **Coarse-to-Fine**: เริ่มจากช่วงค่ากว้างๆ แล้วค่อยๆ ปรับให้ละเอียดขึ้น

3. **Early Stopping**: หยุดการเทรนเมื่อ validation loss ไม่ลดลง

4. **Learning Rate Scheduling**: ปรับ learning rate ระหว่างการเทรน

## ข้อเสนอแนะ

1. **เริ่มจากค่ามาตรฐาน**: ใช้ค่าที่แนะนำในเอกสารหรืองานวิจัยที่เกี่ยวข้อง

2. **ทำความเข้าใจผลกระทบ**: ศึกษาว่า hyperparameter แต่ละตัวส่งผลต่อโมเดลอย่างไร

3. **จดบันทึก**: เก็บบันทึกทุกการทดลองและผลลัพธ์

4. **ใช้ visualization**: สร้างกราฟเพื่อดูความสัมพันธ์ระหว่าง hyperparameters และประสิทธิภาพ

## อ้างอิง

- [A Disciplined Approach to Neural Network Hyper-Parameters](https://arxiv.org/abs/1803.09820)
- [Practical Recommendations for Gradient-Based Training of Deep Architectures](https://arxiv.org/abs/1206.5533)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html) 