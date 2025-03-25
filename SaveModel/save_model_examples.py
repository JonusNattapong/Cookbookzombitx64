#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ตัวอย่างการเซฟโมเดล AI แบบต่างๆ
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from datetime import datetime
import pickle
import gzip
import json
from cryptography.fernet import Fernet


# สร้างโฟลเดอร์สำหรับเก็บโมเดล
os.makedirs("saved_models", exist_ok=True)


# สร้างโมเดลตัวอย่างง่ายๆ ใน PyTorch
class SimpleModel(nn.Module):
    def __init__(self, input_size=10, hidden_size=20, output_size=1):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


def train_model(epochs=5):
    """ฟังก์ชันเทรนโมเดลอย่างง่าย"""
    model = SimpleModel()
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    
    # สร้างข้อมูลตัวอย่าง
    X = torch.randn(100, 10)
    y = torch.randn(100, 1)
    
    # เทรนโมเดล
    loss_history = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())
        print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}')
    
    return model, optimizer, epoch+1, loss_history


def example_1_save_weights_only():
    """1. เซฟเฉพาะน้ำหนัก (Weights Only)"""
    print("\n### 1. เซฟเฉพาะน้ำหนัก (Weights Only) ###")
    model, _, _, _ = train_model()
    
    # เซฟเฉพาะน้ำหนัก
    weights_path = "saved_models/model_weights.pth"
    torch.save(model.state_dict(), weights_path)
    print(f"บันทึกน้ำหนักไปที่: {weights_path}")
    
    # โหลดน้ำหนัก
    loaded_model = SimpleModel()  # ต้องสร้างโมเดลเปล่าก่อน
    loaded_model.load_state_dict(torch.load(weights_path))
    loaded_model.eval()
    print("โหลดน้ำหนักสำเร็จ!")
    
    # ตรวจสอบว่าโมเดลทำงานได้
    with torch.no_grad():
        test_input = torch.randn(1, 10)
        original_output = model(test_input)
        loaded_output = loaded_model(test_input)
        print(f"ผลลัพธ์ต้นฉบับ: {original_output.item():.4f}")
        print(f"ผลลัพธ์หลังโหลด: {loaded_output.item():.4f}")
        print(f"ผลลัพธ์ตรงกัน: {torch.allclose(original_output, loaded_output)}")


def example_2_save_full_model():
    """2. เซฟทั้งโมเดล (Weights + Architecture)"""
    print("\n### 2. เซฟทั้งโมเดล (Weights + Architecture) ###")
    model, _, _, _ = train_model()
    
    # เซฟทั้งโมเดล
    full_model_path = "saved_models/full_model.pth"
    torch.save(model, full_model_path)
    print(f"บันทึกโมเดลทั้งหมดไปที่: {full_model_path}")
    
    # โหลดโมเดล
    loaded_model = torch.load(full_model_path)
    loaded_model.eval()
    print("โหลดโมเดลสำเร็จ!")
    
    # ตรวจสอบว่าโมเดลทำงานได้
    with torch.no_grad():
        test_input = torch.randn(1, 10)
        original_output = model(test_input)
        loaded_output = loaded_model(test_input)
        print(f"ผลลัพธ์ต้นฉบับ: {original_output.item():.4f}")
        print(f"ผลลัพธ์หลังโหลด: {loaded_output.item():.4f}")
        print(f"ผลลัพธ์ตรงกัน: {torch.allclose(original_output, loaded_output)}")


def example_3_save_checkpoint():
    """3. เซฟ Checkpoint (Weights + Optimizer + Epoch)"""
    print("\n### 3. เซฟ Checkpoint (Weights + Optimizer + Epoch) ###")
    model, optimizer, epoch, loss_history = train_model()
    
    # เซฟ checkpoint
    checkpoint_path = "saved_models/checkpoint.pth"
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss_history[-1]
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"บันทึก Checkpoint ไปที่: {checkpoint_path}")
    
    # โหลด checkpoint
    loaded_model = SimpleModel()
    loaded_optimizer = optim.SGD(loaded_model.parameters(), lr=0.01)
    
    checkpoint = torch.load(checkpoint_path)
    loaded_model.load_state_dict(checkpoint['model_state_dict'])
    loaded_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    
    print("โหลด Checkpoint สำเร็จ!")
    print(f"Epoch ล่าสุด: {start_epoch}")
    print(f"Loss ล่าสุด: {loss:.4f}")
    
    # ตรวจสอบว่าโมเดลทำงานได้
    with torch.no_grad():
        test_input = torch.randn(1, 10)
        original_output = model(test_input)
        loaded_output = loaded_model(test_input)
        print(f"ผลลัพธ์ต้นฉบับ: {original_output.item():.4f}")
        print(f"ผลลัพธ์หลังโหลด: {loaded_output.item():.4f}")
        print(f"ผลลัพธ์ตรงกัน: {torch.allclose(original_output, loaded_output)}")
        
    # เทรนต่อได้
    print("เทรนต่ออีก 2 Epochs...")
    criterion = nn.MSELoss()
    X = torch.randn(100, 10)
    y = torch.randn(100, 1)
    
    for epoch in range(start_epoch, start_epoch + 2):
        loaded_optimizer.zero_grad()
        outputs = loaded_model(X)
        loss = criterion(outputs, y)
        loss.backward()
        loaded_optimizer.step()
        print(f'Epoch {epoch}/{start_epoch+2}, Loss: {loss.item():.4f}')


def example_6_save_as_onnx():
    """6. เซฟเป็น ONNX (Open Neural Network Exchange)"""
    print("\n### 6. เซฟเป็น ONNX (Open Neural Network Exchange) ###")
    try:
        model, _, _, _ = train_model()
        
        # เซฟเป็น ONNX
        onnx_path = "saved_models/model.onnx"
        dummy_input = torch.randn(1, 10)
        torch.onnx.export(model, dummy_input, onnx_path)
        print(f"บันทึกโมเดลเป็น ONNX ไปที่: {onnx_path}")
        
        # โหลด ONNX (ต้องมี onnxruntime)
        try:
            import onnxruntime as ort
            session = ort.InferenceSession(onnx_path)
            
            input_name = session.get_inputs()[0].name
            output_name = session.get_outputs()[0].name
            
            # ทำนายด้วย ONNX model
            onnx_input = {input_name: dummy_input.numpy()}
            onnx_output = session.run([output_name], onnx_input)[0]
            
            # เปรียบเทียบกับต้นฉบับ
            original_output = model(dummy_input).detach().numpy()
            print(f"ผลลัพธ์ต้นฉบับ: {original_output[0][0]:.4f}")
            print(f"ผลลัพธ์จาก ONNX: {onnx_output[0][0]:.4f}")
            print(f"ผลลัพธ์ใกล้เคียงกัน: {np.allclose(original_output, onnx_output)}")
            
        except ImportError:
            print("ไม่พบ onnxruntime ให้ติดตั้งด้วย: pip install onnxruntime")
            
    except Exception as e:
        print(f"เกิดข้อผิดพลาด: {e}")


def example_7_save_as_pickle():
    """7. เซฟเป็น Pickle (สำหรับโมเดลเล็กๆ)"""
    print("\n### 7. เซฟเป็น Pickle (สำหรับโมเดลเล็กๆ) ###")
    model, _, _, _ = train_model()
    
    # เซฟเป็น Pickle
    pickle_path = "saved_models/model.pkl"
    with open(pickle_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"บันทึกโมเดลเป็น Pickle ไปที่: {pickle_path}")
    
    # โหลด Pickle
    with open(pickle_path, 'rb') as f:
        loaded_model = pickle.load(f)
    loaded_model.eval()
    print("โหลดโมเดลจาก Pickle สำเร็จ!")
    
    # ตรวจสอบว่าโมเดลทำงานได้
    with torch.no_grad():
        test_input = torch.randn(1, 10)
        original_output = model(test_input)
        loaded_output = loaded_model(test_input)
        print(f"ผลลัพธ์ต้นฉบับ: {original_output.item():.4f}")
        print(f"ผลลัพธ์หลังโหลด: {loaded_output.item():.4f}")
        print(f"ผลลัพธ์ตรงกัน: {torch.allclose(original_output, loaded_output)}")


def example_best_practice_naming():
    """ตัวอย่างการตั้งชื่อไฟล์ที่ดี"""
    print("\n### ตัวอย่างการตั้งชื่อไฟล์ที่ดี ###")
    model, _, _, _ = train_model()
    
    # ใส่ข้อมูลสำคัญในชื่อไฟล์
    version = 1
    accuracy = 0.95
    model_name = f'model_v{version}_acc{accuracy:.2f}_{datetime.now():%Y%m%d}.pth'
    model_path = os.path.join("saved_models", model_name)
    
    torch.save(model.state_dict(), model_path)
    print(f"บันทึกโมเดลไปที่: {model_path}")


def example_save_with_metadata():
    """ตัวอย่างการบันทึกข้อมูลเพิ่มเติม"""
    print("\n### ตัวอย่างการบันทึกข้อมูลเพิ่มเติม ###")
    model, _, _, loss_history = train_model()
    
    # บันทึกข้อมูลเพิ่มเติม
    model_info = {
        'model_state': model.state_dict(),
        'hyperparameters': {
            'learning_rate': 0.01,
            'batch_size': 32,
            'epochs': 5
        },
        'metrics': {
            'accuracy': 0.95,
            'loss': loss_history[-1]
        },
        'date_trained': str(datetime.now()),
        'dataset_info': 'ข้อมูลสร้างแบบสุ่ม 100 ตัวอย่าง'
    }
    
    model_path = "saved_models/model_with_metadata.pth"
    torch.save(model_info, model_path)
    print(f"บันทึกโมเดลพร้อมข้อมูลเพิ่มเติมไปที่: {model_path}")
    
    # โหลดโมเดลพร้อมข้อมูลเพิ่มเติม
    loaded_info = torch.load(model_path)
    print("โหลดโมเดลพร้อมข้อมูลเพิ่มเติมสำเร็จ!")
    print(f"วันที่เทรน: {loaded_info['date_trained']}")
    print(f"Learning Rate: {loaded_info['hyperparameters']['learning_rate']}")
    print(f"Accuracy: {loaded_info['metrics']['accuracy']}")
    
    # โหลดน้ำหนักโมเดล
    loaded_model = SimpleModel()
    loaded_model.load_state_dict(loaded_info['model_state'])
    loaded_model.eval()
    print("โหลดน้ำหนักสำเร็จ!")


def example_save_compressed():
    """ตัวอย่างการบันทึกแบบบีบอัด"""
    print("\n### ตัวอย่างการบันทึกแบบบีบอัด ###")
    model, _, _, _ = train_model()
    
    # บันทึกแบบบีบอัด
    compressed_path = "saved_models/model_compressed.pgz"
    with gzip.open(compressed_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"บันทึกโมเดลแบบบีบอัดไปที่: {compressed_path}")
    
    # โหลดไฟล์ที่ถูกบีบอัด
    with gzip.open(compressed_path, 'rb') as f:
        loaded_model = pickle.load(f)
    loaded_model.eval()
    print("โหลดโมเดลจากไฟล์บีบอัดสำเร็จ!")
    
    # ตรวจสอบว่าโมเดลทำงานได้
    with torch.no_grad():
        test_input = torch.randn(1, 10)
        original_output = model(test_input)
        loaded_output = loaded_model(test_input)
        print(f"ผลลัพธ์ต้นฉบับ: {original_output.item():.4f}")
        print(f"ผลลัพธ์หลังโหลด: {loaded_output.item():.4f}")
        print(f"ผลลัพธ์ตรงกัน: {torch.allclose(original_output, loaded_output)}")


def example_encrypt_model():
    """ตัวอย่างการเข้ารหัสโมเดล"""
    print("\n### ตัวอย่างการเข้ารหัสโมเดล ###")
    try:
        model, _, _, _ = train_model()
        
        # สร้างคีย์
        key = Fernet.generate_key()
        cipher_suite = Fernet(key)
        print(f"สร้างคีย์: {key.decode()[:20]}...")
        
        # เข้ารหัสและบันทึก
        model_bytes = pickle.dumps(model)
        encrypted_model = cipher_suite.encrypt(model_bytes)
        
        encrypted_path = "saved_models/model_encrypted.bin"
        with open(encrypted_path, 'wb') as file:
            file.write(encrypted_model)
        print(f"บันทึกโมเดลแบบเข้ารหัสไปที่: {encrypted_path}")
        
        # ถอดรหัสและโหลด
        with open(encrypted_path, 'rb') as file:
            encrypted_model = file.read()
        decrypted_model = cipher_suite.decrypt(encrypted_model)
        loaded_model = pickle.loads(decrypted_model)
        loaded_model.eval()
        print("ถอดรหัสและโหลดโมเดลสำเร็จ!")
        
        # ตรวจสอบว่าโมเดลทำงานได้
        with torch.no_grad():
            test_input = torch.randn(1, 10)
            original_output = model(test_input)
            loaded_output = loaded_model(test_input)
            print(f"ผลลัพธ์ต้นฉบับ: {original_output.item():.4f}")
            print(f"ผลลัพธ์หลังโหลด: {loaded_output.item():.4f}")
            print(f"ผลลัพธ์ตรงกัน: {torch.allclose(original_output, loaded_output)}")
            
    except ImportError:
        print("ไม่พบ cryptography ให้ติดตั้งด้วย: pip install cryptography")
    except Exception as e:
        print(f"เกิดข้อผิดพลาด: {e}")


def example_model_checkpoint_class():
    """ตัวอย่างการใช้ ModelCheckpoint"""
    print("\n### ตัวอย่างการใช้ ModelCheckpoint ###")
    
    class ModelCheckpoint:
        def __init__(self, filepath, monitor='loss', save_best_only=True):
            self.filepath = filepath
            self.monitor = monitor
            self.save_best_only = save_best_only
            self.best = float('inf') if monitor == 'loss' else float('-inf')
        
        def save(self, model, current_value):
            if self.save_best_only:
                improved = ((self.monitor == 'loss' and current_value < self.best) or
                          (self.monitor != 'loss' and current_value > self.best))
                if improved:
                    self.best = current_value
                    torch.save(model.state_dict(), self.filepath)
                    return True, self.best
                return False, self.best
            else:
                torch.save(model.state_dict(), self.filepath)
                return True, current_value
    
    # สร้างโมเดลและเตรียมข้อมูล
    model = SimpleModel()
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    
    X = torch.randn(100, 10)
    y = torch.randn(100, 1)
    
    # สร้าง checkpoint callback
    checkpoint = ModelCheckpoint(filepath="saved_models/best_model.pth")
    
    # เทรนโมเดล
    epochs = 10
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        
        # ลองบันทึกโมเดล
        saved, best_value = checkpoint.save(model, loss.item())
        status = "บันทึกแล้ว" if saved else "ไม่บันทึก (ไม่ดีกว่าเดิม)"
        print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}, สถานะ: {status}, ค่าที่ดีที่สุด: {best_value:.4f}')


def main():
    """ฟังก์ชันหลัก"""
    print("=" * 50)
    print("ตัวอย่างการเซฟโมเดล AI แบบต่างๆ")
    print("=" * 50)
    
    examples = [
        example_1_save_weights_only,
        example_2_save_full_model,
        example_3_save_checkpoint,
        example_6_save_as_onnx,
        example_7_save_as_pickle,
        example_best_practice_naming,
        example_save_with_metadata,
        example_save_compressed,
        example_encrypt_model,
        example_model_checkpoint_class
    ]
    
    for example in examples:
        try:
            example()
            print("-" * 50)
        except Exception as e:
            print(f"เกิดข้อผิดพลาดในตัวอย่าง {example.__name__}: {e}")
            print("-" * 50)
    
    print("\nเสร็จสิ้นการแสดงตัวอย่าง")
    print(f"ไฟล์โมเดลทั้งหมดถูกบันทึกไว้ในโฟลเดอร์: {os.path.abspath('saved_models')}")


if __name__ == "__main__":
    main() 