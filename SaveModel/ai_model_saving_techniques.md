# วิธีการเซฟโมเดล AI แบบต่างๆ

## 1. เซฟเฉพาะน้ำหนัก (Weights Only)
- **แนวคิด**: เซฟแค่พารามิเตอร์ (weights) ของโมเดล โดยไม่รวมโครงสร้าง (architecture)  
- **วิธีการ**:  
  - **PyTorch**:  
    ```python
    torch.save(model.state_dict(), 'model_weights.pth')
    ```
  - **TensorFlow/Keras**:  
    ```python
    model.save_weights('model_weights.h5')
    ```
- **ผลลัพธ์**: ได้ไฟล์ `.pth` หรือ `.h5` ที่เก็บน้ำหนัก  
- **ข้อดี**:  
  - ไฟล์เล็ก เพราะไม่มีโครงสร้าง  
  - เหมาะกับงานที่โครงสร้างโมเดลไม่เปลี่ยนบ่อย  
- **ข้อเสีย**:  
  - ต้องเขียนโค้ดกำหนดโครงสร้างโมเดลเองตอนโหลด  
  - ไม่สะดวกถ้าต้องแชร์ให้คนอื่น  
- **การโหลด**:  
  - PyTorch:  
    ```python
    model.load_state_dict(torch.load('model_weights.pth'))
    ```
  - TensorFlow/Keras:  
    ```python
    model.load_weights('model_weights.h5')
    ```
- **เหมาะกับ**: งานวิจัยหรือทีมที่แชร์โค้ดกัน  

## 2. เซฟทั้งโมเดล (Weights + Architecture)
- **แนวคิด**: เซฟทั้งน้ำหนักและโครงสร้างโมเดลในไฟล์เดียว  
- **วิธีการ**:  
  - **PyTorch**:  
    ```python
    torch.save(model, 'full_model.pth')
    ```
  - **TensorFlow/Keras**:  
    ```python
    model.save('full_model.h5')
    ```
- **ผลลัพธ์**: ได้ไฟล์ `.pth` หรือ `.h5` ที่เก็บทั้งน้ำหนักและโครงสร้าง  
- **ข้อดี**:  
  - โหลดง่าย ไม่ต้องกำหนดโครงสร้างใหม่  
  - เหมาะกับ deployment หรือแชร์ให้คนอื่น  
- **ข้อเสีย**:  
  - ไฟล์ใหญ่กว่า  
  - อาจมีปัญหา compatibility ถ้า framework version ไม่ตรง  
- **การโหลด**:  
  - PyTorch:  
    ```python
    model = torch.load('full_model.pth')
    ```
  - TensorFlow/Keras:  
    ```python
    from tensorflow.keras.models import load_model
    model = load_model('full_model.h5')
    ```
- **เหมาะกับ**: การ deploy หรือส่งโมเดลให้คนอื่น  

## 3. เซฟ Checkpoint (Weights + Optimizer + Epoch)
- **แนวคิด**: เซฟสถานะของการเทรน (น้ำหนัก, optimizer state, epoch) เพื่อให้เทรนต่อได้  
- **วิธีการ**:  
  - **PyTorch**:  
    ```python
    checkpoint = {
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'epoch': epoch
    }
    torch.save(checkpoint, 'checkpoint.pth')
    ```
  - **TensorFlow**:  
    ```python
    checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
    checkpoint.save('checkpoint')
    ```
- **ผลลัพธ์**: ได้ไฟล์ `.pth` หรือ checkpoint files ที่เก็บน้ำหนัก, optimizer state, และ epoch  
- **ข้อดี**:  
  - เทรนต่อได้จากจุดที่หยุด  
  - เหมาะกับการเทรนยาวๆ ที่อาจสะดุด  
- **ข้อเสีย**:  
  - ไฟล์อาจใหญ่ขึ้นถ้ามี optimizer state  
  - ต้องจัดการ optimizer ตอนโหลด  
- **การโหลด**:  
  - PyTorch:  
    ```python
    checkpoint = torch.load('checkpoint.pth')
    model.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    start_epoch = checkpoint['epoch']
    ```
  - TensorFlow:  
    ```python
    checkpoint.restore('checkpoint')
    ```
- **เหมาะกับ**: การเทรนที่ใช้เวลานาน  

## 4. เซฟแบบ Hugging Face (Weights + Config + Tokenizer)
- **แนวคิด**: เซฟโมเดลทั้งหมด (น้ำหนัก, โครงสร้าง, config, tokenizer) ในรูปแบบที่ Hugging Face ใช้  
- **วิธีการ**:  
  ```python
  from transformers import AutoModel, AutoTokenizer

  # สมมติว่า model และ tokenizer คือโมเดลที่คุณเทรน
  model.save_pretrained("my_model")
  tokenizer.save_pretrained("my_model")
  ```
- **ผลลัพธ์**: ได้โฟลเดอร์ `my_model` ที่มีไฟล์:  
  - `config.json`: เก็บโครงสร้างและการตั้งค่า  
  - `pytorch_model.bin` หรือ `model.safetensors`: เก็บน้ำหนัก  
  - `tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json`: เก็บ tokenizer  
  - `generation_config.json` (ถ้ามี): เก็บการตั้งค่าสำหรับการ generate  
- **ข้อดี**:  
  - โหลดง่ายด้วย Hugging Face library  
  - รองรับ tokenizer และ config ทำให้ใช้งานได้ทันที  
- **ข้อเสีย**:  
  - ไฟล์เยอะ อาจดูซับซ้อน  
  - เหมาะกับโมเดลที่ใช้ Hugging Face เท่านั้น  
- **การโหลด**:  
  ```python
  from transformers import AutoModel, AutoTokenizer
  model = AutoModel.from_pretrained("my_model")
  tokenizer = AutoTokenizer.from_pretrained("my_model")
  ```
- **เหมาะกับ**: โมเดลภาษา (NLP) หรือ generative AI  

## 5. เซฟแบบ Sharded (แยกไฟล์น้ำหนักเป็นส่วนๆ)
- **แนวคิด**: แยกน้ำหนักโมเดลเป็นไฟล์เล็กๆ (shards) เพื่อให้จัดการง่าย  
- **วิธีการ**:  
  - **Hugging Face**:  
    ```python
    model.save_pretrained("my_model", max_shard_size="5GB")
    ```
- **ผลลัพธ์**: ได้ไฟล์น้ำหนักแยก เช่น:  
  - `model-00001-of-00002.safetensors`  
  - `model-00002-of-00002.safetensors`  
  - และ `model.safetensors.index.json` เพื่อบอกว่าแต่ละ shard เก็บอะไร  
- **ข้อดี**:  
  - เหมาะกับโมเดลใหญ่ (เช่น 10 GB+)  
  - โหลดทีละส่วนได้ ไม่ต้องใช้ RAM เยอะ  
- **ข้อเสีย**:  
  - ไฟล์เยอะ อาจสับสน  
  - ต้องใช้ library ที่รองรับ (เช่น Hugging Face)  
- **การโหลด**: เหมือนวิธี Hugging Face ทั่วไป  
- **เหมาะกับ**: โมเดลขนาดใหญ่ เช่น GPT, LLaMA  

## 6. เซฟเป็น ONNX (Open Neural Network Exchange)
- **แนวคิด**: แปลงโมเดลเป็นรูปแบบ ONNX ซึ่งเป็น standard กลางที่รองรับหลาย framework  
- **วิธีการ**:  
  - **PyTorch**:  
    ```python
    import torch.onnx
    dummy_input = torch.randn(1, input_size)  # ตัวอย่าง input
    torch.onnx.export(model, dummy_input, 'model.onnx')
    ```
  - **TensorFlow**: ใช้ `tf2onnx`  
- **ผลลัพธ์**: ได้ไฟล์ `model.onnx`  
- **ข้อดี**:  
  - ใช้ข้าม framework ได้ (เช่น จาก PyTorch ไป TensorFlow)  
  - เหมาะกับ deployment บน platform ต่างๆ  
- **ข้อเสีย**:  
  - ต้องมี dummy input  
  - บาง operation อาจไม่รองรับ  
- **การโหลด**:  
  - ใช้ ONNX Runtime:  
    ```python
    import onnxruntime as ort
    session = ort.InferenceSession('model.onnx')
    ```
- **เหมาะกับ**: การ deploy ข้าม platform  

## 7. เซฟเป็น Pickle (สำหรับโมเดลเล็กๆ)
- **แนวคิด**: ใช้ Python's `pickle` เซฟโมเดลทั้งหมดเป็นไฟล์ binary  
- **วิธีการ**:  
  ```python
  import pickle
  with open('model.pkl', 'wb') as f:
      pickle.dump(model, f)
  ```
- **ผลลัพธ์**: ได้ไฟล์ `model.pkl`  
- **ข้อดี**:  
  - ง่ายสุดๆ ไม่ต้องใช้ API พิเศษ  
  - เหมาะกับโมเดลเล็กๆ หรือ prototyping  
- **ข้อเสีย**:  
  - ไม่ปลอดภัย (security risk) ถ้าโหลดจากแหล่งที่ไม่น่าไว้ใจ  
  - ไม่เหมาะกับโมเดลใหญ่  
- **การโหลด**:  
  ```python
  with open('model.pkl', 'rb') as f:
      model = pickle.load(f)
  ```
- **เหมาะกับ**: งานทดลองหรือโปรเจ็คส่วนตัว  

## 8. เซฟเป็น JSON + Weights
- **แนวคิด**: เซฟโครงสร้างเป็น JSON และน้ำหนักแยกเป็นไฟล์  
- **วิธีการ**:  
  - **Keras**:  
    ```python
    with open('model_architecture.json', 'w') as f:
        f.write(model.to_json())
    model.save_weights('model_weights.h5')
    ```
- **ผลลัพธ์**: ได้ไฟล์ `model_architecture.json` และ `model_weights.h5`  
- **ข้อดี**:  
  - JSON อ่านง่าย (human-readable)  
  - เหมาะกับการ debug หรือวิเคราะห์โมเดล  
- **ข้อเสีย**:  
  - ไม่สะดวกกับ framework อื่นนอกจาก Keras  
- **การโหลด**:  
  ```python
  from tensorflow.keras.models import model_from_json
  with open('model_architecture.json', 'r') as f:
      model = model_from_json(f.read())
  model.load_weights('model_weights.h5')
  ```
- **เหมาะกับ**: งานที่ต้องการตรวจสอบโครงสร้าง  

## 9. เซฟเป็น TensorFlow SavedModel
- **แนวคิด**: เซฟโมเดลในรูปแบบ TensorFlow SavedModel ซึ่งเหมาะกับ deployment  
- **วิธีการ**:  
  ```python
  tf.saved_model.save(model, 'saved_model')
  ```
- **ผลลัพธ์**: ได้โฟลเดอร์ `saved_model` ที่มีไฟล์:  
  - `saved_model.pb`: เก็บโครงสร้างและ metadata  
  - โฟลเดอร์ `variables`: เก็บน้ำหนัก  
- **ข้อดี**:  
  - เหมาะกับ deployment (เช่น TensorFlow Serving)  
  - รองรับการใช้งานใน production  
- **ข้อเสีย**:  
  - ใช้ได้ดีกับ TensorFlow เท่านั้น  
- **การโหลด**:  
  ```python
  model = tf.saved_model.load('saved_model')
  ```
- **เหมาะกับ**: การ deploy ใน production  

## สรุป: วิธีเซฟแบบไหนดี?
- **ถ้าต้องการเทรนต่อ**: เซฟ checkpoint (วิธี 3)  
- **ถ้าต้องการ deploy**:  
  - ใช้ Hugging Face (วิธี 4) ถ้าเป็นโมเดลภาษา  
  - ใช้ ONNX (วิธี 6) หรือ TensorFlow SavedModel (วิธี 9) ถ้าต้องการ cross-platform  
- **ถ้าโมเดลใหญ่**: ใช้ sharded saving (วิธี 5)  
- **ถ้าต้องการง่ายๆ**: เซฟ weights เฉยๆ (วิธี 1) หรือใช้ pickle (วิธี 7) 