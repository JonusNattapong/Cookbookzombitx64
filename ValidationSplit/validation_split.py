#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
การแบ่งข้อมูลเป็น Train และ Validation Set
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score

# สร้างโฟลเดอร์สำหรับเก็บผลลัพธ์
os.makedirs("plots", exist_ok=True)

# กำหนด seed เพื่อให้ผลลัพธ์คงที่
torch.manual_seed(42)
np.random.seed(42)

class SimpleClassifier(nn.Module):
    def __init__(self, input_size=20, hidden_size=50, output_size=2):
        super(SimpleClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

class ClassificationDataset(data.Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def generate_data(n_samples=1000, n_features=20, n_classes=2):
    """สร้างข้อมูลจำลองสำหรับการจำแนกประเภท"""
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_classes=n_classes,
        n_informative=n_features//2,
        random_state=42
    )
    return X, y

def train_without_validation(X, y, epochs=50, batch_size=32, lr=0.01):
    """ฝึกโมเดลโดยไม่มีการแบ่ง validation set"""
    print("=" * 50)
    print("การเทรนโดยไม่มี Validation Split")
    print("=" * 50)
    
    # สร้างโมเดลและกำหนดค่าต่างๆ
    model = SimpleClassifier()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # สร้าง dataset และ dataloader
    dataset = ClassificationDataset(X, y)
    dataloader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # บันทึกประวัติการเทรน
    train_losses = []
    
    # เทรนโมเดล
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        
        for inputs, targets in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        avg_epoch_loss = epoch_loss / len(dataloader)
        train_losses.append(avg_epoch_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_epoch_loss:.4f}")
    
    # ทดสอบประสิทธิภาพบนข้อมูลฝึกสอน
    model.eval()
    with torch.no_grad():
        outputs = model(torch.FloatTensor(X))
        _, predicted = torch.max(outputs.data, 1)
        accuracy = accuracy_score(y, predicted.numpy())
        
    print(f"การเทรนเสร็จสิ้น")
    print(f"ความแม่นยำบนข้อมูลฝึกสอน: {accuracy:.4f}")
    print(f"Loss สุดท้าย: {train_losses[-1]:.4f}")
    
    # พล็อตกราฟ loss
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.title('การเทรนโดยไม่มี Validation Split')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('plots/train_only_loss.png')
    
    return model, train_losses

def train_with_validation(X, y, val_ratio=0.2, epochs=50, batch_size=32, lr=0.01):
    """ฝึกโมเดลโดยมีการแบ่ง validation set"""
    print("\n" + "=" * 50)
    print(f"การเทรนโดยมี Validation Split (สัดส่วน: {val_ratio*100:.0f}%)")
    print("=" * 50)
    
    # สร้างโมเดลและกำหนดค่าต่างๆ
    model = SimpleClassifier()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # สร้าง dataset
    dataset = ClassificationDataset(X, y)
    
    # แบ่งข้อมูลเป็น train และ validation
    dataset_size = len(dataset)
    val_size = int(val_ratio * dataset_size)
    train_size = dataset_size - val_size
    
    train_dataset, val_dataset = data.random_split(
        dataset, [train_size, val_size]
    )
    
    # สร้าง dataloader
    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = data.DataLoader(val_dataset, batch_size=batch_size)
    
    # บันทึกประวัติการเทรน
    train_losses = []
    val_losses = []
    
    # สำหรับตรวจสอบ early stopping
    best_val_loss = float('inf')
    patience = 5  # จำนวน epoch ที่รอก่อนหยุด
    patience_counter = 0
    best_model_state = None
    
    # เทรนโมเดล
    for epoch in range(epochs):
        # โหมดเทรน
        model.train()
        train_epoch_loss = 0
        
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_epoch_loss += loss.item()
            
        avg_train_epoch_loss = train_epoch_loss / len(train_loader)
        train_losses.append(avg_train_epoch_loss)
        
        # โหมดประเมินผล
        model.eval()
        val_epoch_loss = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_epoch_loss += loss.item()
                
        avg_val_epoch_loss = val_epoch_loss / len(val_loader)
        val_losses.append(avg_val_epoch_loss)
        
        # แสดงผลทุก 10 epoch
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], "
                  f"Train Loss: {avg_train_epoch_loss:.4f}, "
                  f"Val Loss: {avg_val_epoch_loss:.4f}")
        
        # ตรวจสอบ early stopping
        if avg_val_epoch_loss < best_val_loss:
            best_val_loss = avg_val_epoch_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"Early stopping ที่ epoch {epoch+1}")
            model.load_state_dict(best_model_state)
            break
    
    # ประเมินประสิทธิภาพบน validation set
    model.eval()
    val_predictions = []
    val_targets = []
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            val_predictions.extend(predicted.numpy())
            val_targets.extend(targets.numpy())
    
    val_accuracy = accuracy_score(val_targets, val_predictions)
    
    print(f"การเทรนเสร็จสิ้น")
    print(f"ความแม่นยำบน validation set: {val_accuracy:.4f}")
    print(f"Train Loss สุดท้าย: {train_losses[-1]:.4f}")
    print(f"Validation Loss สุดท้าย: {val_losses[-1]:.4f}")
    
    # พล็อตกราฟ loss
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('การเทรนโดยมี Validation Split')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('plots/train_val_loss.png')
    
    return model, train_losses, val_losses

def compare_performance(with_val_losses, without_val_losses, val_losses=None):
    """เปรียบเทียบประสิทธิภาพระหว่างการเทรนแบบมีและไม่มี validation"""
    plt.figure(figsize=(12, 6))
    plt.plot(without_val_losses, label='Train Loss (ไม่มี Validation)')
    plt.plot(with_val_losses, label='Train Loss (มี Validation)')
    
    if val_losses:
        plt.plot(val_losses, label='Validation Loss')
    
    plt.title('เปรียบเทียบ Loss ระหว่างวิธีการฝึกสอนต่างๆ')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('plots/comparison_loss.png')
    plt.close()

def learning_curve_analysis(X, y, val_ratio=0.2, epochs=50, batch_size=32, lr=0.01):
    """วิเคราะห์ learning curve ที่ขนาดข้อมูลต่างๆ"""
    print("\n" + "=" * 50)
    print("การวิเคราะห์ Learning Curve")
    print("=" * 50)
    
    # สร้าง dataset
    dataset = ClassificationDataset(X, y)
    
    # แบ่งข้อมูลเป็น train และ validation
    dataset_size = len(dataset)
    val_size = int(val_ratio * dataset_size)
    train_size = dataset_size - val_size
    
    train_dataset, val_dataset = data.random_split(
        dataset, [train_size, val_size]
    )
    
    # กำหนดสัดส่วนของข้อมูลที่จะใช้
    train_fractions = [0.1, 0.25, 0.5, 0.75, 1.0]
    train_sizes = [int(f * train_size) for f in train_fractions]
    
    train_scores = []
    val_scores = []
    
    for size in train_sizes:
        print(f"\nทดลองใช้ข้อมูลฝึกสอน {size} ตัวอย่าง")
        
        # สุ่มเลือกข้อมูลตามขนาดที่กำหนด
        subset_indices = torch.randperm(train_size)[:size].tolist()
        subset_train = data.Subset(train_dataset, subset_indices)
        
        # สร้าง dataloader
        train_loader = data.DataLoader(subset_train, batch_size=batch_size, shuffle=True)
        val_loader = data.DataLoader(val_dataset, batch_size=batch_size)
        
        # สร้างโมเดลและกำหนดค่าต่างๆ
        model = SimpleClassifier()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        # เทรนโมเดล
        for epoch in range(epochs):
            model.train()
            for inputs, targets in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
        
        # ประเมินผลบน training set
        model.eval()
        train_correct = 0
        train_total = 0
        
        with torch.no_grad():
            for inputs, targets in train_loader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                train_total += targets.size(0)
                train_correct += (predicted == targets).sum().item()
        
        train_accuracy = train_correct / train_total
        train_scores.append(train_accuracy)
        
        # ประเมินผลบน validation set
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                val_total += targets.size(0)
                val_correct += (predicted == targets).sum().item()
        
        val_accuracy = val_correct / val_total
        val_scores.append(val_accuracy)
        
        print(f"ความแม่นยำบนข้อมูลฝึกสอน: {train_accuracy:.4f}")
        print(f"ความแม่นยำบน validation set: {val_accuracy:.4f}")
    
    # พล็อต learning curve
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_scores, 'o-', label='Training Accuracy')
    plt.plot(train_sizes, val_scores, 'o-', label='Validation Accuracy')
    plt.title('Learning Curve')
    plt.xlabel('จำนวนตัวอย่างในชุดข้อมูลฝึกสอน')
    plt.ylabel('Accuracy')
    plt.legend(loc='best')
    plt.grid(True)
    plt.savefig('plots/learning_curve.png')
    plt.close()
    
    return train_sizes, train_scores, val_scores

def main():
    # พารามิเตอร์
    n_samples = 1000
    n_features = 20
    n_classes = 2
    epochs = 50
    batch_size = 32
    learning_rate = 0.01
    validation_ratio = 0.2
    
    # สร้างข้อมูล
    X, y = generate_data(n_samples, n_features, n_classes)
    
    # ฝึกโมเดลโดยไม่มี validation
    model_without_val, train_losses_without_val = train_without_validation(
        X, y, epochs, batch_size, learning_rate
    )
    
    # ฝึกโมเดลโดยมี validation
    model_with_val, train_losses_with_val, val_losses = train_with_validation(
        X, y, validation_ratio, epochs, batch_size, learning_rate
    )
    
    # เปรียบเทียบประสิทธิภาพ
    compare_performance(train_losses_with_val, train_losses_without_val, val_losses)
    
    # วิเคราะห์ learning curve
    train_sizes, train_scores, val_scores = learning_curve_analysis(
        X, y, validation_ratio, epochs, batch_size, learning_rate
    )
    
    print("\n" + "=" * 50)
    print("สรุปผลการทดลอง")
    print("=" * 50)
    print(f"1. การเทรนโดยไม่มี validation:")
    print(f"   - สามารถฝึกสอนได้เร็วกว่า เนื่องจากใช้ข้อมูลน้อยกว่า")
    print(f"   - Loss สุดท้าย: {train_losses_without_val[-1]:.4f}")
    print(f"   - ไม่สามารถประเมินความสามารถในการทำนายข้อมูลใหม่ได้")
    
    print(f"\n2. การเทรนโดยมี validation:")
    print(f"   - ใช้ข้อมูลฝึกสอนน้อยกว่า ({(1-validation_ratio)*100:.0f}% ของข้อมูลทั้งหมด)")
    print(f"   - Train Loss สุดท้าย: {train_losses_with_val[-1]:.4f}")
    print(f"   - Validation Loss สุดท้าย: {val_losses[-1]:.4f}")
    print(f"   - สามารถตรวจสอบ overfitting และ underfitting ได้")
    print(f"   - สามารถใช้ early stopping เพื่อหยุดการเทรนเมื่อโมเดลเริ่ม overfit")
    
    print(f"\n3. การวิเคราะห์ Learning Curve:")
    print(f"   - เมื่อมีข้อมูลเพิ่มขึ้น ความแม่นยำบน validation set จะ:")
    
    # ตรวจสอบว่า accuracy เพิ่มขึ้นหรือลดลงเมื่อข้อมูลเพิ่ม
    if val_scores[-1] > val_scores[0]:
        print(f"     * เพิ่มขึ้น: จาก {val_scores[0]:.4f} เป็น {val_scores[-1]:.4f}")
        print(f"     * แสดงว่าการเพิ่มข้อมูลฝึกสอนช่วยให้โมเดลเรียนรู้ได้ดีขึ้น")
    else:
        print(f"     * ไม่เพิ่มขึ้นอย่างมีนัยสำคัญ: จาก {val_scores[0]:.4f} เป็น {val_scores[-1]:.4f}")
        print(f"     * แสดงว่าการเพิ่มข้อมูลอาจไม่ช่วยให้โมเดลดีขึ้น ควรปรับปรุงโมเดลหรือคุณภาพข้อมูลแทน")
    
    # ตรวจสอบ gap ระหว่าง train กับ validation accuracy
    gap_start = abs(train_scores[0] - val_scores[0])
    gap_end = abs(train_scores[-1] - val_scores[-1])
    
    if gap_end > gap_start:
        print(f"     * ช่องว่างระหว่าง training และ validation accuracy เพิ่มขึ้น:")
        print(f"       จาก {gap_start:.4f} เป็น {gap_end:.4f}")
        print(f"     * แสดงว่าโมเดลอาจกำลัง overfit เมื่อข้อมูลเพิ่มขึ้น")
    else:
        print(f"     * ช่องว่างระหว่าง training และ validation accuracy ลดลง:")
        print(f"       จาก {gap_start:.4f} เป็น {gap_end:.4f}")
        print(f"     * แสดงว่าโมเดลสามารถ generalize ได้ดีขึ้นเมื่อมีข้อมูลเพิ่มขึ้น")

if __name__ == "__main__":
    main() 