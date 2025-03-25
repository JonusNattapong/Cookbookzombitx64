#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Cross-Domain Knowledge Distillation
การถ่ายโอนความรู้ระหว่าง domain ที่แตกต่างกัน (จากภาพไปข้อความ)
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, random_split
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
from PIL import Image
import time
import json
from tqdm import tqdm

# สร้างโฟลเดอร์สำหรับเก็บผลลัพธ์
os.makedirs("models", exist_ok=True)
os.makedirs("plots", exist_ok=True)

# กำหนด seed เพื่อให้ผลลัพธ์คงที่
torch.manual_seed(42)
np.random.seed(42)

class TeacherImageModel(nn.Module):
    """โมเดลครู (Teacher) สำหรับ image domain โดยใช้ pretrained ResNet18"""
    def __init__(self, num_classes=10):
        super(TeacherImageModel, self).__init__()
        self.model = models.resnet18(pretrained=True)
        # ปรับ fully connected layer สุดท้ายให้เหมาะกับจำนวนคลาส
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        
    def forward(self, x):
        # ส่งคืนทั้ง logits และ features จากชั้นก่อนสุดท้าย
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        
        x = self.model.avgpool(x)
        features = torch.flatten(x, 1)
        logits = self.model.fc(features)
        return logits, features

class StudentTextModel(nn.Module):
    """โมเดลนักเรียน (Student) สำหรับ text domain"""
    def __init__(self, vocab_size, embedding_dim, hidden_dim, feature_dim, num_classes):
        super(StudentTextModel, self).__init__()
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # LSTM layer
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        # FC layer เพื่อให้ได้ feature dimension เดียวกับครู
        self.fc_feature = nn.Linear(hidden_dim, feature_dim)
        # Output layer
        self.fc_out = nn.Linear(feature_dim, num_classes)
        
    def forward(self, x):
        # x มีขนาด (batch_size, sequence_length)
        embedded = self.embedding(x)  # (batch_size, sequence_length, embedding_dim)
        lstm_out, _ = self.lstm(embedded)  # (batch_size, sequence_length, hidden_dim)
        # ใช้ output ตัวสุดท้ายของ sequence
        last_hidden = lstm_out[:, -1, :]  # (batch_size, hidden_dim)
        # สร้าง features ที่มี dimension เดียวกับ teacher
        features = self.fc_feature(last_hidden)  # (batch_size, feature_dim)
        # สร้าง logits สำหรับการจำแนก
        logits = self.fc_out(features)  # (batch_size, num_classes)
        return logits, features

# ข้อมูลจำลองสำหรับ demo
class SyntheticMultiModalDataset(Dataset):
    """
    ข้อมูลจำลองที่มีทั้งภาพและข้อความ
    - image: ภาพสีขนาด 224x224
    - text: ลำดับของตัวเลขที่แทน token (จำลอง)
    - label: คลาสของข้อมูล (0-9)
    """
    def __init__(self, num_samples=1000, image_size=224, text_length=20, vocab_size=1000, num_classes=10):
        self.num_samples = num_samples
        self.image_size = image_size
        self.text_length = text_length
        self.vocab_size = vocab_size
        self.num_classes = num_classes
        
        # สร้างข้อมูลจำลอง
        self.images = torch.randn(num_samples, 3, image_size, image_size)
        self.texts = torch.randint(0, vocab_size, (num_samples, text_length))
        self.labels = torch.randint(0, num_classes, (num_samples,))
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return {
            'image': self.images[idx],
            'text': self.texts[idx],
            'label': self.labels[idx]
        }

def train_teacher_model(model, dataloader, criterion, optimizer, device, num_epochs=5):
    """
    ฝึกสอนโมเดลครู (Teacher) ด้วยข้อมูลภาพ
    """
    model.train()
    history = {'train_loss': [], 'train_acc': []}
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        predictions = []
        true_labels = []
        
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            
            # Forward pass
            logits, _ = model(images)
            loss = criterion(logits, labels)
            
            # Backward และ optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            
            # เก็บข้อมูลสำหรับคำนวณความแม่นยำ
            _, preds = torch.max(logits, 1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
        
        epoch_loss = running_loss / len(dataloader.dataset)
        epoch_acc = accuracy_score(true_labels, predictions)
        
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc)
        
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")
    
    return history

def train_student_with_distillation(teacher_model, student_model, dataloader, criterion_ce, criterion_mse, 
                                   optimizer, device, num_epochs=10, alpha=0.5, temperature=2.0):
    """
    ฝึกสอนโมเดลนักเรียน (Student) ด้วย knowledge distillation
    - alpha: น้ำหนักระหว่าง loss ของคลาสที่แท้จริงกับ loss ของการกลั่น (0-1)
    - temperature: อุณหภูมิสำหรับการกลั่น (Softmax temperature)
    """
    teacher_model.eval()
    student_model.train()
    
    history = {
        'train_loss': [], 
        'distillation_loss': [], 
        'ce_loss': [], 
        'train_acc': []
    }
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        running_kd_loss = 0.0
        running_ce_loss = 0.0
        predictions = []
        true_labels = []
        
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            texts = batch['text'].to(device)
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            
            # 1. ใช้โมเดลครูทำนายจากภาพ
            with torch.no_grad():
                teacher_logits, teacher_features = teacher_model(images)
                
            # 2. ให้โมเดลนักเรียนทำนายจากข้อความ
            student_logits, student_features = student_model(texts)
            
            # 3. คำนวณ loss
            # 3.1 Cross-entropy loss กับ label จริง
            ce_loss = criterion_ce(student_logits, labels)
            
            # 3.2 Knowledge distillation loss (soft targets)
            # ใช้ feature distillation
            feature_loss = criterion_mse(student_features, teacher_features)
            
            # 3.3 Soft target distillation (logits)
            T = temperature
            soft_targets = nn.functional.softmax(teacher_logits / T, dim=1)
            log_probs = nn.functional.log_softmax(student_logits / T, dim=1)
            soft_targets_loss = -torch.sum(soft_targets * log_probs) / soft_targets.size(0)
            
            # รวม loss
            distillation_loss = feature_loss + soft_targets_loss
            loss = alpha * ce_loss + (1 - alpha) * distillation_loss
            
            # Backward และ optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * texts.size(0)
            running_kd_loss += distillation_loss.item() * texts.size(0)
            running_ce_loss += ce_loss.item() * texts.size(0)
            
            # เก็บข้อมูลสำหรับคำนวณความแม่นยำ
            _, preds = torch.max(student_logits, 1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
        
        epoch_loss = running_loss / len(dataloader.dataset)
        epoch_kd_loss = running_kd_loss / len(dataloader.dataset)
        epoch_ce_loss = running_ce_loss / len(dataloader.dataset)
        epoch_acc = accuracy_score(true_labels, predictions)
        
        history['train_loss'].append(epoch_loss)
        history['distillation_loss'].append(epoch_kd_loss)
        history['ce_loss'].append(epoch_ce_loss)
        history['train_acc'].append(epoch_acc)
        
        print(f"Epoch {epoch+1}/{num_epochs} - "
              f"Loss: {epoch_loss:.4f}, CE Loss: {epoch_ce_loss:.4f}, "
              f"KD Loss: {epoch_kd_loss:.4f}, Accuracy: {epoch_acc:.4f}")
    
    return history

def train_student_without_distillation(student_model, dataloader, criterion, 
                                      optimizer, device, num_epochs=10):
    """
    ฝึกสอนโมเดลนักเรียน (Student) โดยไม่ใช้ knowledge distillation
    """
    student_model.train()
    history = {'train_loss': [], 'train_acc': []}
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        predictions = []
        true_labels = []
        
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            texts = batch['text'].to(device)
            labels = batch['label'].to(device)
            
            # Forward pass
            logits, _ = student_model(texts)
            loss = criterion(logits, labels)
            
            # Backward และ optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * texts.size(0)
            
            # เก็บข้อมูลสำหรับคำนวณความแม่นยำ
            _, preds = torch.max(logits, 1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
        
        epoch_loss = running_loss / len(dataloader.dataset)
        epoch_acc = accuracy_score(true_labels, predictions)
        
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc)
        
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")
    
    return history

def evaluate_model(model, dataloader, criterion, device, is_text_model=True):
    """
    ประเมินประสิทธิภาพของโมเดล
    """
    model.eval()
    running_loss = 0.0
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # เลือกใช้ input ตามประเภทของโมเดล
            if is_text_model:
                inputs = batch['text'].to(device)
            else:
                inputs = batch['image'].to(device)
                
            labels = batch['label'].to(device)
            
            # Forward pass
            logits, _ = model(inputs)
            loss = criterion(logits, labels)
            
            running_loss += loss.item() * inputs.size(0)
            
            # เก็บข้อมูลสำหรับคำนวณความแม่นยำ
            _, preds = torch.max(logits, 1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    
    eval_loss = running_loss / len(dataloader.dataset)
    eval_acc = accuracy_score(true_labels, predictions)
    
    return eval_loss, eval_acc

def plot_training_history(history_dict, title, save_path):
    """
    สร้างกราฟแสดงผลการฝึกสอน
    """
    plt.figure(figsize=(12, 5))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    for key, values in history_dict.items():
        if 'loss' in key:
            plt.plot(values, label=key)
    
    plt.title(f'{title} - Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    for key, values in history_dict.items():
        if 'acc' in key:
            plt.plot(values, label=key)
    
    plt.title(f'{title} - Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def compare_models(teacher_acc, student_with_kd_acc, student_without_kd_acc, epochs, save_path):
    """
    เปรียบเทียบประสิทธิภาพระหว่างโมเดลต่างๆ
    """
    plt.figure(figsize=(10, 6))
    
    plt.plot(range(1, len(teacher_acc) + 1), teacher_acc, 'o-', label='Teacher (Image) Model')
    plt.plot(range(1, len(student_with_kd_acc) + 1), student_with_kd_acc, 's-', label='Student (Text) with Distillation')
    plt.plot(range(1, len(student_without_kd_acc) + 1), student_without_kd_acc, '^-', label='Student (Text) without Distillation')
    
    plt.title('Model Accuracy Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def run_cross_domain_knowledge_distillation():
    """
    ดำเนินการทดลอง Cross-Domain Knowledge Distillation
    """
    # พารามิเตอร์
    num_classes = 10
    vocab_size = 1000
    embedding_dim = 100
    hidden_dim = 128
    feature_dim = 512  # ต้องตรงกับ feature dimension ของ ResNet18
    batch_size = 32
    teacher_epochs = 5
    student_epochs = 10
    
    # เลือกอุปกรณ์ในการคำนวณ
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # สร้างข้อมูลจำลอง
    print("Creating synthetic multimodal dataset...")
    full_dataset = SyntheticMultiModalDataset(num_samples=2000)
    
    # แบ่งข้อมูลเป็นชุดฝึกสอนและตรวจสอบ
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # สร้างโมเดล
    print("Creating teacher (image) and student (text) models...")
    teacher_model = TeacherImageModel(num_classes=num_classes).to(device)
    student_model_with_kd = StudentTextModel(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        feature_dim=feature_dim,
        num_classes=num_classes
    ).to(device)
    
    student_model_without_kd = StudentTextModel(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        feature_dim=feature_dim,
        num_classes=num_classes
    ).to(device)
    
    # กำหนด Loss function และ Optimizer
    criterion_ce = nn.CrossEntropyLoss()
    criterion_mse = nn.MSELoss()
    
    teacher_optimizer = optim.Adam(teacher_model.parameters(), lr=0.001)
    student_with_kd_optimizer = optim.Adam(student_model_with_kd.parameters(), lr=0.001)
    student_without_kd_optimizer = optim.Adam(student_model_without_kd.parameters(), lr=0.001)
    
    # 1. ฝึกสอนโมเดลครู (Teacher) ด้วยข้อมูลภาพ
    print("\n" + "="*40)
    print("Training Teacher (Image) Model...")
    print("="*40)
    
    teacher_history = train_teacher_model(
        model=teacher_model,
        dataloader=train_loader,
        criterion=criterion_ce,
        optimizer=teacher_optimizer,
        device=device,
        num_epochs=teacher_epochs
    )
    
    # บันทึกโมเดลครู
    torch.save(teacher_model.state_dict(), 'models/teacher_image_model.pth')
    
    # 2. ฝึกสอนโมเดลนักเรียน (Student) ด้วย Knowledge Distillation
    print("\n" + "="*40)
    print("Training Student (Text) Model with Knowledge Distillation...")
    print("="*40)
    
    student_with_kd_history = train_student_with_distillation(
        teacher_model=teacher_model,
        student_model=student_model_with_kd,
        dataloader=train_loader,
        criterion_ce=criterion_ce,
        criterion_mse=criterion_mse,
        optimizer=student_with_kd_optimizer,
        device=device,
        num_epochs=student_epochs,
        alpha=0.5,
        temperature=2.0
    )
    
    # บันทึกโมเดลนักเรียนที่ใช้ Knowledge Distillation
    torch.save(student_model_with_kd.state_dict(), 'models/student_text_model_with_kd.pth')
    
    # 3. ฝึกสอนโมเดลนักเรียน (Student) แบบปกติ (ไม่ใช้ Knowledge Distillation)
    print("\n" + "="*40)
    print("Training Student (Text) Model without Knowledge Distillation...")
    print("="*40)
    
    student_without_kd_history = train_student_without_distillation(
        student_model=student_model_without_kd,
        dataloader=train_loader,
        criterion=criterion_ce,
        optimizer=student_without_kd_optimizer,
        device=device,
        num_epochs=student_epochs
    )
    
    # บันทึกโมเดลนักเรียนที่ไม่ได้ใช้ Knowledge Distillation
    torch.save(student_model_without_kd.state_dict(), 'models/student_text_model_without_kd.pth')
    
    # 4. ประเมินและเปรียบเทียบโมเดล
    print("\n" + "="*40)
    print("Evaluating Models on Validation Set...")
    print("="*40)
    
    teacher_val_loss, teacher_val_acc = evaluate_model(
        model=teacher_model,
        dataloader=val_loader,
        criterion=criterion_ce,
        device=device,
        is_text_model=False
    )
    
    student_with_kd_val_loss, student_with_kd_val_acc = evaluate_model(
        model=student_model_with_kd,
        dataloader=val_loader,
        criterion=criterion_ce,
        device=device,
        is_text_model=True
    )
    
    student_without_kd_val_loss, student_without_kd_val_acc = evaluate_model(
        model=student_model_without_kd,
        dataloader=val_loader,
        criterion=criterion_ce,
        device=device,
        is_text_model=True
    )
    
    print(f"\nTeacher (Image) Model - Validation Loss: {teacher_val_loss:.4f}, Accuracy: {teacher_val_acc:.4f}")
    print(f"Student (Text) with KD - Validation Loss: {student_with_kd_val_loss:.4f}, Accuracy: {student_with_kd_val_acc:.4f}")
    print(f"Student (Text) without KD - Validation Loss: {student_without_kd_val_loss:.4f}, Accuracy: {student_without_kd_val_acc:.4f}")
    
    # 5. สร้างกราฟแสดงผลการฝึกสอน
    plot_training_history(
        teacher_history,
        'Teacher (Image) Model Training',
        'plots/teacher_training_history.png'
    )
    
    plot_training_history(
        student_with_kd_history,
        'Student (Text) Model with Knowledge Distillation',
        'plots/student_with_kd_training_history.png'
    )
    
    plot_training_history(
        student_without_kd_history,
        'Student (Text) Model without Knowledge Distillation',
        'plots/student_without_kd_training_history.png'
    )
    
    # 6. เปรียบเทียบประสิทธิภาพระหว่างโมเดล
    compare_models(
        teacher_acc=teacher_history['train_acc'],
        student_with_kd_acc=student_with_kd_history['train_acc'],
        student_without_kd_acc=student_without_kd_history['train_acc'],
        epochs=max(teacher_epochs, student_epochs),
        save_path='plots/model_accuracy_comparison.png'
    )
    
    # 7. สรุปผลการทดลอง
    results = {
        'teacher_val_accuracy': teacher_val_acc,
        'student_with_kd_val_accuracy': student_with_kd_val_acc,
        'student_without_kd_val_accuracy': student_without_kd_val_acc,
        'accuracy_improvement': student_with_kd_val_acc - student_without_kd_val_acc,
        'improvement_percentage': (student_with_kd_val_acc - student_without_kd_val_acc) / student_without_kd_val_acc * 100
    }
    
    with open('results_summary.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    print("\n" + "="*40)
    print("Results Summary:")
    print("="*40)
    print(f"Teacher (Image) Model Accuracy: {teacher_val_acc:.4f}")
    print(f"Student (Text) with Knowledge Distillation Accuracy: {student_with_kd_val_acc:.4f}")
    print(f"Student (Text) without Knowledge Distillation Accuracy: {student_without_kd_val_acc:.4f}")
    print(f"Absolute Improvement with Knowledge Distillation: {results['accuracy_improvement']:.4f}")
    print(f"Relative Improvement: {results['improvement_percentage']:.2f}%")
    
    if results['accuracy_improvement'] > 0:
        print("\nConclusion: Cross-Domain Knowledge Distillation successfully improved the performance of the text model.")
    else:
        print("\nConclusion: In this experiment, Cross-Domain Knowledge Distillation did not improve the performance of the text model.")

if __name__ == "__main__":
    run_cross_domain_knowledge_distillation() 