# วิธีการบันทึกโมเดล AI

คู่มือนี้จะอธิบายวิธีการบันทึกโมเดล AI ในรูปแบบต่างๆ เพื่อนำไปใช้งานในภายหลัง

## 1. การบันทึกโมเดล PyTorch

### 1.1 บันทึกทั้งโมเดล
```python
import torch

# บันทึกทั้งโมเดล
torch.save(model, 'model.pth')

# โหลดโมเดล
model = torch.load('model.pth')
model.eval()  # สำหรับการทำนาย
```

### 1.2 บันทึกเฉพาะ State Dict
```python
# บันทึก state dict
torch.save(model.state_dict(), 'model_state_dict.pth')

# โหลด state dict
model = YourModelClass()
model.load_state_dict(torch.load('model_state_dict.pth'))
model.eval()
```

### 1.3 บันทึกพร้อม Checkpoint
```python
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss
}
torch.save(checkpoint, 'checkpoint.pth')

# โหลด checkpoint
model = YourModelClass()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
checkpoint = torch.load('checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']
```

## 2. การบันทึกโมเดล TensorFlow/Keras

### 2.1 บันทึกทั้งโมเดล
```python
from tensorflow import keras

# บันทึกทั้งโมเดล
model.save('model.h5')

# โหลดโมเดล
model = keras.models.load_model('model.h5')
```

### 2.2 บันทึกเฉพาะ Weights
```python
# บันทึก weights
model.save_weights('model_weights.h5')

# โหลด weights
model = YourModelClass()
model.load_weights('model_weights.h5')
```

### 2.3 บันทึกในรูปแบบ SavedModel
```python
# บันทึกในรูปแบบ SavedModel
model.save('saved_model_dir')

# โหลดโมเดล
model = keras.models.load_model('saved_model_dir')
```

## 3. การบันทึกโมเดล Scikit-learn

### 3.1 ใช้ Pickle
```python
import pickle

# บันทึกโมเดล
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)

# โหลดโมเดล
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)
```

### 3.2 ใช้ Joblib (แนะนำสำหรับไฟล์ขนาดใหญ่)
```python
from joblib import dump, load

# บันทึกโมเดล
dump(model, 'model.joblib')

# โหลดโมเดล
model = load('model.joblib')
```

## 4. การบันทึกโมเดล ONNX

### 4.1 แปลงและบันทึกโมเดล PyTorch เป็น ONNX
```python
import torch.onnx

# แปลงและบันทึกเป็น ONNX
dummy_input = torch.randn(1, input_size)
torch.onnx.export(model, dummy_input, "model.onnx")

# โหลดโมเดล ONNX
import onnxruntime
session = onnxruntime.InferenceSession("model.onnx")
```

### 4.2 แปลงและบันทึกโมเดล Keras เป็น ONNX
```python
import keras2onnx

# แปลงเป็น ONNX
onnx_model = keras2onnx.convert_keras(model)

# บันทึกโมเดล
with open("model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())
```

## 5. ข้อแนะนำในการบันทึกโมเดล

1. การตั้งชื่อไฟล์
```python
# ใส่ข้อมูลสำคัญในชื่อไฟล์
model_name = f'model_v{version}_acc{accuracy:.2f}_{datetime.now():%Y%m%d}.pth'
```

2. การบันทึกข้อมูลเพิ่มเติม
```python
model_info = {
    'model_state': model.state_dict(),
    'hyperparameters': {
        'learning_rate': lr,
        'batch_size': batch_size,
        'epochs': epochs
    },
    'metrics': {
        'accuracy': accuracy,
        'loss': loss
    },
    'date_trained': str(datetime.now()),
    'dataset_info': dataset_info
}
torch.save(model_info, 'model_complete_info.pth')
```

3. การบันทึกอัตโนมัติระหว่างเทรน
```python
class ModelCheckpoint:
    def __init__(self, filepath, monitor='loss', save_best_only=True):
        self.filepath = filepath
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.best = float('inf') if monitor == 'loss' else float('-inf')
    
    def save(self, model, current_value):
        if self.save_best_only:
            if ((self.monitor == 'loss' and current_value < self.best) or
                (self.monitor != 'loss' and current_value > self.best)):
                self.best = current_value
                torch.save(model.state_dict(), self.filepath)
        else:
            torch.save(model.state_dict(), self.filepath)
```

## 6. การจัดการพื้นที่จัดเก็บ

1. การบีบอัดไฟล์
```python
import gzip
import pickle

# บันทึกแบบบีบอัด
with gzip.open('model_compressed.pgz', 'wb') as f:
    pickle.dump(model, f)

# โหลดไฟล์ที่ถูกบีบอัด
with gzip.open('model_compressed.pgz', 'rb') as f:
    model = pickle.load(f)
```

2. การลบไฟล์เก่า
```python
import os
from glob import glob

def cleanup_old_models(directory, keep_last_n=5):
    files = glob(os.path.join(directory, 'model_*.pth'))
    if len(files) > keep_last_n:
        files.sort(key=os.path.getmtime)
        for f in files[:-keep_last_n]:
            os.remove(f)
```

## 7. ความปลอดภัย

1. การเข้ารหัสโมเดล
```python
from cryptography.fernet import Fernet
import pickle

# สร้างคีย์
key = Fernet.generate_key()
cipher_suite = Fernet(key)

# เข้ารหัสและบันทึก
model_bytes = pickle.dumps(model)
encrypted_model = cipher_suite.encrypt(model_bytes)
with open('model_encrypted.bin', 'wb') as file:
    file.write(encrypted_model)

# ถอดรหัสและโหลด
with open('model_encrypted.bin', 'rb') as file:
    encrypted_model = file.read()
decrypted_model = cipher_suite.decrypt(encrypted_model)
model = pickle.loads(decrypted_model)
```

## 8. การตรวจสอบความถูกต้อง

```python
def verify_saved_model(original_model, saved_model_path, test_input):
    # ทำนายด้วยโมเดลต้นฉบับ
    original_prediction = original_model(test_input)
    
    # โหลดและทำนายด้วยโมเดลที่บันทึก
    loaded_model = torch.load(saved_model_path)
    loaded_prediction = loaded_model(test_input)
    
    # เปรียบเทียบผลลัพธ์
    return torch.allclose(original_prediction, loaded_prediction)
```

## ข้อควรระวัง

1. เวอร์ชันของไลบรารี
   - บันทึกเวอร์ชันของไลบรารีที่ใช้
   - ทดสอบการโหลดโมเดลในสภาพแวดล้อมที่ต่างกัน

2. ขนาดไฟล์
   - ตรวจสอบขนาดไฟล์ก่อนบันทึก
   - พิจารณาใช้การบีบอัดสำหรับไฟล์ขนาดใหญ่

3. ความเข้ากันได้
   - ทดสอบการโหลดโมเดลในแพลตฟอร์มต่างๆ
   - บันทึกข้อมูลการกำหนดค่าที่จำเป็น 