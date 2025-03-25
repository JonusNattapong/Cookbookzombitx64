#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ตัวอย่างการใช้งานโมดูล EpochTrainer สำหรับการเทรนโมเดลแบบ Epoch-Based
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from dotenv import load_dotenv
from cookbookzombitx64.training.epoch_based_training import EpochTrainer, MetricTracker

# โหลดค่าตัวแปรสภาพแวดล้อม
load_dotenv()

class SimpleNN(nn.Module):
    """โมเดลเครือข่ายประสาทเทียมอย่างง่ายสำหรับการจำแนกตัวเลข MNIST"""
    
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.flatten = nn.Flatten()
        self.layers = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 10)
        )
        
    def forward(self, x):
        x = self.flatten(x)
        logits = self.layers(x)
        return logits

def main():
    """ฟังก์ชันหลักสำหรับตัวอย่างการใช้งาน"""
    
    print("=== ตัวอย่างการใช้งาน EpochTrainer ===")
    
    # ตรวจสอบอุปกรณ์ที่ใช้ในการคำนวณ
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"ใช้อุปกรณ์: {device}")
    
    # โหลดชุดข้อมูล MNIST
    print("กำลังโหลดชุดข้อมูล MNIST...")
    mnist_train = MNIST(root='./data', train=True, download=True, transform=ToTensor())
    mnist_test = MNIST(root='./data', train=False, download=True, transform=ToTensor())
    
    # สร้าง DataLoader
    train_loader = DataLoader(mnist_train, batch_size=64, shuffle=True)
    test_loader = DataLoader(mnist_test, batch_size=64, shuffle=False)
    
    print(f"ชุดข้อมูลฝึก: {len(mnist_train)} ตัวอย่าง ({len(train_loader)} แบตช์)")
    print(f"ชุดข้อมูลทดสอบ: {len(mnist_test)} ตัวอย่าง ({len(test_loader)} แบตช์)")
    
    # สร้างโมเดล
    model = SimpleNN()
    
    # กำหนดฟังก์ชันสำหรับคำนวณค่าความสูญเสีย (loss)
    loss_fn = nn.CrossEntropyLoss()
    
    # กำหนด optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # กำหนด learning rate scheduler (ตัวอย่าง: ReduceLROnPlateau)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=3, verbose=True
    )
    
    # กำหนดเมทริกสำหรับวัดประสิทธิภาพ
    metrics = {
        'accuracy': MetricTracker.accuracy,
        'precision': MetricTracker.precision,
        'recall': MetricTracker.recall,
        'f1_score': MetricTracker.f1_score
    }
    
    # สร้างออบเจ็กต์ EpochTrainer
    trainer = EpochTrainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        scheduler=scheduler
    )
    
    # กำหนดไดเรกทอรีสำหรับบันทึก checkpoint
    checkpoint_dir = "./checkpoints/mnist"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # เทรนโมเดล
    print("\nเริ่มการเทรนโมเดล...")
    history = trainer.fit(
        train_loader=train_loader,
        num_epochs=5,  # เพื่อความรวดเร็วในการทดสอบ ใช้เพียง 5 epoch
        val_loader=test_loader,
        metrics=metrics,
        checkpoint_dir=checkpoint_dir,
        early_stopping=10,  # หยุดการเทรนหลังจาก 10 epoch ถ้า val_loss ไม่ลดลง
        verbose=True
    )
    
    print("\nการเทรนเสร็จสิ้น!")
    print(f"ประวัติ loss สุดท้าย: train_loss={history['train_loss'][-1]:.4f}, val_loss={history['val_loss'][-1]:.4f}")
    print(f"ความแม่นยำสุดท้าย: train_accuracy={history['metrics']['accuracy']['train'][-1]:.4f}, val_accuracy={history['metrics']['accuracy']['val'][-1]:.4f}")
    
    # โหลด checkpoint ของโมเดลที่ดีที่สุด
    print("\nโหลดโมเดลที่ดีที่สุด...")
    trainer.load_checkpoint(os.path.join(checkpoint_dir, 'best_model.pth'))
    
    # ประเมินผลโมเดลกับชุดข้อมูลทดสอบ
    print("\nประเมินผลโมเดลกับชุดข้อมูลทดสอบ:")
    test_results = trainer.validate(test_loader, metrics)
    print(f"Test Loss: {test_results['loss']:.4f}")
    print(f"Test Accuracy: {test_results['accuracy']:.4f}")
    print(f"Test Precision: {test_results['precision']:.4f}")
    print(f"Test Recall: {test_results['recall']:.4f}")
    print(f"Test F1 Score: {test_results['f1_score']:.4f}")
    
    print("\nการทดสอบเสร็จสิ้น!")

if __name__ == "__main__":
    main() 