# Cookbookzombitx64

โปรเจกต์นี้มุ่งเน้นการพัฒนาโมเดลเอไอ โดยมีเป้าหมายให้ทุกคนสามารถเข้าถึงและใช้งานได้ง่าย

## โครงสร้างโปรเจกต์
- **LoadModel**: โมดูลสำหรับโหลดโมเดล
- **Cross-Domain Knowledge Distillation**: การถ่ายทอดความรู้ข้ามโดเมน
- **MCTS Neural Networks**: เครือข่ายประสาทเทียมที่ใช้ MCTS
- **Manual Hyperparameter Tuning**: การปรับแต่งไฮเปอร์พารามิเตอร์ด้วยตนเอง
- **Validation Split**: การแบ่งข้อมูลสำหรับการตรวจสอบ
- **Energy-Efficient Training Scheduler**: ตัวจัดการการฝึกที่ประหยัดพลังงาน
- **SaveModel**: โมดูลสำหรับบันทึกโมเดล
- **Dataset Generator**: เครื่องมือสร้างชุดข้อมูล
- **Epoch Based Training**: การฝึกที่อิงตามยุค
- **Gradient Descent Variants**: รวมถึง Vanilla, Stochastic, และ MiniBatch Gradient Descent

## วิธีการเทรนโมเดล
### LLM Finetuning (การปรับแต่งโมเดลภาษาขนาดใหญ่)
- **LLM SFT**: การปรับแต่งโมเดลด้วยเทคนิค Supervised Fine-Tuning
- **LLM ORPO**: การปรับแต่งโมเดลด้วยเทคนิค ORPO
- **LLM Generic**: การปรับแต่งโมเดลทั่วไป
- **LLM DPO**: การปรับแต่งโมเดลด้วยเทคนิค Direct Preference Optimization
- **LLM Reward**: การปรับแต่งโมเดลโดยใช้รางวัลเป็นตัวชี้นำ

### VLM Finetuning (การปรับแต่งโมเดลภาพและภาษา)
- **VLM Captioning**: การสร้างคำอธิบายภาพ
- **VLM VQA**: การตอบคำถามจากภาพ (Visual Question Answering)

### Sentence Transformers
- **ST Pair**: การทำงานกับคู่ประโยค
- **ST Pair Classification**: การจำแนกคู่ประโยค
- **ST Pair Scoring**: การให้คะแนนคู่ประโยค
- **ST Triplet**: การทำงานกับชุดประโยคแบบสามส่วน
- **ST Question Answering**: การตอบคำถามด้วยประโยค

### Other Text Tasks (งานประมวลผลข้อความอื่นๆ)
- **Text Classification**: การจำแนกประเภทข้อความ
- **Text Regression**: การถดถอยข้อความ
- **Extractive Question Answering**: การตอบคำถามแบบสกัดข้อมูล
- **Sequence To Sequence**: การแปลงลำดับไปสู่ลำดับ
- **Token Classification**: การจำแนกประเภทโทเค็น

### Image Tasks (งานประมวลผลภาพ)
- **Image Classification**: การจำแนกประเภทภาพ
- **Image Scoring/Regression**: การให้คะแนนหรือการถดถอยภาพ
- **Object Detection**: การตรวจจับวัตถุในภาพ

### Tabular Tasks (งานประมวลผลข้อมูลตาราง)
- **Tabular Classification**: การจำแนกประเภทข้อมูลตาราง
- **Tabular Regression**: การถดถอยข้อมูลตาราง

## การติดตั้ง
ดูไฟล์ `requirements.txt` สำหรับการติดตั้ง dependencies

## การใช้งาน
คำแนะนำการใช้งานเบื้องต้น

## การสนับสนุน
ข้อมูลการติดต่อสำหรับการสนับสนุน

## ลิขสิทธิ์
ข้อมูลลิขสิทธิ์ในไฟล์ `LICENSE`