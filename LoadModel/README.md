# โค้ดสำหรับโหลดโมเดลจาก Hugging Face

โค้ดนี้ใช้สำหรับโหลดโมเดลจาก Hugging Face มาเก็บไว้ที่เครื่อง Local เพื่อใช้งานในภายหลังโดยไม่ต้องโหลดใหม่ทุกครั้ง

## ความต้องการ

โค้ดนี้ต้องการแพ็คเกจต่อไปนี้:
- transformers
- torch
- python-dotenv

ถ้ายังไม่มีแพ็คเกจเหล่านี้ ให้ติดตั้งด้วยคำสั่ง:

```
pip install transformers torch python-dotenv
```

## การใช้งาน

### วิธีที่ 1: ใช้งานผ่าน Command Line

```
python load_model.py --model MODEL_NAME [--type TYPE] [--save_dir SAVE_DIR]
```

ตัวอย่าง:
```
# โหลด BERT model และ tokenizer
python load_model.py --model bert-base-uncased

# โหลดเฉพาะ model
python load_model.py --model gpt2 --type model

# โหลดเฉพาะ tokenizer
python load_model.py --model roberta-base --type tokenizer

# ระบุไดเร็กทอรีที่จะบันทึก
python load_model.py --model distilbert-base-uncased --save_dir ./my_models/distilbert
```

### วิธีที่ 2: ใช้งานใน Python Code

```python
from load_model import load_model_from_huggingface, load_model_from_local

# โหลดโมเดลจาก Hugging Face
model, tokenizer = load_model_from_huggingface("bert-base-uncased", model_type="both")

# โหลดโมเดลที่บันทึกไว้ใน Local
model, tokenizer = load_model_from_local("./bert-base-uncased")
```

## ตัวอย่างการทำงาน

ไฟล์ `example.py` ให้ตัวอย่างการใช้งานทั้งหมด:

1. การโหลดโมเดลจาก Hugging Face
2. การโหลดโมเดลจาก Local
3. การใช้งานโมเดลและ tokenizer

รันตัวอย่างด้วยคำสั่ง:
```
python example.py
```

## คำอธิบายฟังก์ชัน

### `load_model_from_huggingface(model_name, model_type="model", save_dir=None)`

โหลดโมเดลจาก Hugging Face และบันทึกไว้ที่ Local

#### พารามิเตอร์:
- `model_name` (str): ชื่อโมเดลจาก Hugging Face (เช่น 'bert-base-uncased')
- `model_type` (str): ประเภทของโมเดล ('model', 'tokenizer', หรือ 'both')
- `save_dir` (str): ไดเร็กทอรีที่ต้องการบันทึกโมเดล (ถ้าไม่ระบุจะใช้ชื่อโมเดล)

#### คืนค่า:
- `model`, `tokenizer` หรือทั้งคู่ตามค่า `model_type`

### `load_model_from_local(model_dir, model_type="both")`

โหลดโมเดลที่บันทึกไว้ในเครื่อง Local

#### พารามิเตอร์:
- `model_dir` (str): ไดเร็กทอรีที่บันทึกโมเดลไว้
- `model_type` (str): ประเภทของโมเดล ('model', 'tokenizer', หรือ 'both')

#### คืนค่า:
- `model`, `tokenizer` หรือตัวใดตัวหนึ่งตามค่า `model_type` 