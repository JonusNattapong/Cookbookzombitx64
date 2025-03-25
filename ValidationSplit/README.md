# การแบ่งข้อมูลเป็น Train และ Validation Set

เทคนิคการแบ่งข้อมูลเป็น train set และ validation set เป็นหนึ่งในเทคนิคพื้นฐานที่สำคัญที่สุดสำหรับการฝึกสอนโมเดล machine learning และ deep learning อย่างมีประสิทธิภาพ

## แนวคิด

- แบ่งข้อมูลออกเป็น 2 ส่วน: ส่วนสำหรับการฝึกสอน (training set) และส่วนสำหรับการตรวจสอบ (validation set)
- ใช้ training set ในการเรียนรู้พารามิเตอร์ของโมเดล
- ใช้ validation set ในการประเมินประสิทธิภาพของโมเดลระหว่างการฝึกสอน
- ช่วยในการตัดสินใจว่าควรหยุดการฝึกสอนเมื่อใด หรือควรปรับปรุงโมเดลอย่างไร

## วิธีการ

1. แบ่งข้อมูลเป็น 2 ส่วน โดยทั่วไปจะใช้สัดส่วน:
   - 80% สำหรับ training set
   - 20% สำหรับ validation set

2. ฝึกสอนโมเดลบน training set

3. หลังจากจบแต่ละ epoch:
   - ประเมินผลบน validation set
   - บันทึกค่า loss และ metrics อื่นๆ (เช่น accuracy)

4. ตรวจสอบการ overfitting:
   - ถ้า validation loss เริ่มเพิ่มขึ้นในขณะที่ training loss ยังคงลดลง แสดงว่าโมเดลกำลัง overfit
   - สามารถใช้ early stopping เพื่อหยุดการเทรนเมื่อโมเดลเริ่ม overfit

## เครื่องมือที่ใช้

- `torch.utils.data.random_split`: สำหรับแบ่งข้อมูลใน PyTorch
- `sklearn.model_selection.train_test_split`: สำหรับแบ่งข้อมูลด้วย scikit-learn

## จุดเด่น

1. ช่วยตรวจสอบ overfitting:
   - สามารถดูได้ว่าโมเดลสามารถทำนายข้อมูลที่ไม่เคยเห็นมาก่อนได้ดีเพียงใด

2. ช่วยในการเลือก hyperparameters:
   - ใช้ผลลัพธ์จาก validation set ในการเลือก hyperparameters ที่เหมาะสม

3. ช่วยในการ early stopping:
   - หยุดการเทรนเมื่อโมเดลไม่ได้พัฒนาบน validation set แล้ว

4. ช่วยประหยัดทรัพยากร:
   - ไม่ต้องเทรนจนครบทุก epoch ที่กำหนดไว้ ถ้าโมเดลเริ่ม overfit

## ข้อจำกัด

1. ลดปริมาณข้อมูลฝึกสอน:
   - เมื่อแบ่งส่วนหนึ่งไปเป็น validation set ทำให้ข้อมูลสำหรับการฝึกสอนลดลง

2. อาจเกิดความไม่คงที่:
   - การแบ่งข้อมูลแบบสุ่มอาจทำให้ผลลัพธ์ไม่คงที่ในแต่ละครั้ง

3. ยังไม่ใช่การประเมินผลลัพธ์สุดท้าย:
   - ต้องใช้ test set ที่แยกออกมาสำหรับการประเมินผลลัพธ์สุดท้าย

## การใช้งานในโค้ดตัวอย่าง

โค้ดตัวอย่างนี้แสดงวิธีการใช้ validation split ในการฝึกสอนโมเดล และเปรียบเทียบกับการฝึกสอนแบบไม่มี validation:

1. ติดตั้ง dependencies:
```bash
pip install torch numpy matplotlib scikit-learn
```

2. รันโค้ดตัวอย่าง:
```bash
python validation_split.py
```

3. ผลลัพธ์ที่ได้จะแสดง:
   - กราฟเปรียบเทียบ loss ระหว่างการฝึกสอนแบบมีและไม่มี validation
   - กราฟ learning curve ที่แสดงความสัมพันธ์ระหว่างขนาดข้อมูลและประสิทธิภาพ
   - สรุปผลการเปรียบเทียบระหว่างวิธีการทั้งสอง

## วิธีการวิเคราะห์ผลลัพธ์

1. **Training และ Validation Loss**:
   - ถ้า training loss ลดลงแต่ validation loss เพิ่มขึ้น → โมเดลกำลัง overfit
   - ถ้าทั้ง training และ validation loss ยังลดลงอยู่ → โมเดลยังสามารถเรียนรู้ได้อีก
   - ถ้าทั้ง training และ validation loss ยังสูง → โมเดลอาจจะ underfit

2. **Learning Curve**:
   - ถ้า validation accuracy เพิ่มขึ้นเมื่อข้อมูลเพิ่มขึ้น → ควรเพิ่มข้อมูลฝึกสอน
   - ถ้า gap ระหว่าง training และ validation accuracy กว้างขึ้น → โมเดลกำลัง overfit
   - ถ้า training และ validation accuracy ต่ำทั้งคู่ → โมเดลอาจจะ underfit

## เทคนิคเพิ่มเติม

1. **Cross-Validation**: แบ่งข้อมูลเป็นหลายส่วนและหมุนเวียนการใช้ validation set
2. **Stratified Sampling**: แบ่งข้อมูลโดยรักษาสัดส่วนของ class ให้เท่ากัน
3. **Time-Based Splitting**: สำหรับข้อมูลแบบ time series ควรแบ่งตามช่วงเวลา

## อ้างอิง

- [PyTorch Documentation on Data Loading](https://pytorch.org/docs/stable/data.html)
- [Scikit-learn Documentation on Model Evaluation](https://scikit-learn.org/stable/modules/model_evaluation.html)
- [Overfitting and Underfitting in Machine Learning](https://www.tensorflow.org/tutorials/keras/overfit_and_underfit) 