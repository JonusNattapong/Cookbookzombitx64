#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
การปรับ Hyperparameter แบบ Manual
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
import itertools
import pandas as pd
from tabulate import tabulate
import time

# สร้างโฟลเดอร์สำหรับเก็บผลลัพธ์
os.makedirs("plots", exist_ok=True)
os.makedirs("results", exist_ok=True)

# กำหนด seed เพื่อให้ผลลัพธ์คงที่
torch.manual_seed(42)
np.random.seed(42)

class SimpleClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

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

def train_and_evaluate(model, criterion, optimizer, X_train, y_train, X_val, y_val, 
                       batch_size=32, epochs=50, device='cpu', early_stopping=True, patience=5):
    """
    ฝึกสอนโมเดลและประเมินผล
    
    Returns:
        dict: ผลลัพธ์ประกอบด้วย loss และ metrics ต่างๆ
    """
    # แปลงข้อมูลเป็น tensor
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.LongTensor(y_train).to(device)
    X_val_tensor = torch.FloatTensor(X_val).to(device)
    y_val_tensor = torch.LongTensor(y_val).to(device)
    
    # ประวัติการเทรน
    train_losses = []
    val_losses = []
    val_accuracies = []
    val_f1_scores = []
    
    # สำหรับตรวจสอบ early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    best_epoch = 0
    
    # จับเวลาการเทรน
    start_time = time.time()
    
    # เทรนโมเดล
    n_batches = int(np.ceil(len(X_train) / batch_size))
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        # สร้าง mini-batch
        indices = torch.randperm(len(X_train_tensor))
        
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(X_train_tensor))
            batch_indices = indices[start_idx:end_idx]
            
            X_batch = X_train_tensor[batch_indices]
            y_batch = y_train_tensor[batch_indices]
            
            # Forward pass
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            # Backward และ optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * len(X_batch)
        
        train_loss /= len(X_train_tensor)
        train_losses.append(train_loss)
        
        # ประเมินผลบน validation set
        model.eval()
        with torch.no_grad():
            outputs = model(X_val_tensor)
            val_loss = criterion(outputs, y_val_tensor).item()
            val_losses.append(val_loss)
            
            _, predicted = torch.max(outputs.data, 1)
            val_accuracy = accuracy_score(y_val_tensor.cpu().numpy(), predicted.cpu().numpy())
            val_accuracies.append(val_accuracy)
            
            val_f1 = f1_score(y_val_tensor.cpu().numpy(), predicted.cpu().numpy(), average='macro')
            val_f1_scores.append(val_f1)
        
        # ตรวจสอบ early stopping
        if early_stopping:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = model.state_dict().copy()
                best_epoch = epoch
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                # โหลด best model
                model.load_state_dict(best_model_state)
                print(f"Early stopping ที่ epoch {epoch+1} (best epoch: {best_epoch+1})")
                break
    
    # คำนวณเวลาที่ใช้
    training_time = time.time() - start_time
    
    # สรุปผลลัพธ์
    results = {
        'final_train_loss': train_losses[-1],
        'final_val_loss': val_losses[-1],
        'best_val_loss': min(val_losses),
        'best_epoch': val_losses.index(min(val_losses)) + 1,
        'final_val_accuracy': val_accuracies[-1],
        'best_val_accuracy': max(val_accuracies),
        'final_val_f1': val_f1_scores[-1],
        'best_val_f1': max(val_f1_scores),
        'epochs_completed': len(train_losses),
        'training_time': training_time,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies,
        'val_f1_scores': val_f1_scores
    }
    
    return results

def grid_search_manual(X_train, y_train, X_val, y_val, param_grid, device='cpu'):
    """
    ทำการค้นหา hyperparameters แบบ grid search ด้วยวิธี manual
    
    Args:
        X_train, y_train: ข้อมูลฝึกสอน
        X_val, y_val: ข้อมูลตรวจสอบ
        param_grid: dict ของ hyperparameters และค่าที่จะลอง
        device: อุปกรณ์ที่ใช้ในการคำนวณ
        
    Returns:
        dict: ผลลัพธ์ประกอบด้วย hyperparameters ที่ดีที่สุดและผลลัพธ์
    """
    # สร้างการรวมกันทั้งหมดของ hyperparameters
    param_combinations = list(itertools.product(*param_grid.values()))
    param_names = list(param_grid.keys())
    
    best_accuracy = 0
    best_f1 = 0
    best_params = None
    all_results = []
    
    print(f"เริ่มค้นหา hyperparameters แบบ manual grid search ({len(param_combinations)} การทดลอง)...")
    
    # ลอง hyperparameters แต่ละชุด
    for i, params in enumerate(tqdm(param_combinations, desc="Grid Search Progress")):
        param_dict = dict(zip(param_names, params))
        
        # สร้างโมเดล
        model = SimpleClassifier(
            input_size=X_train.shape[1], 
            hidden_size=param_dict['hidden_size'], 
            num_classes=len(np.unique(y_train))
        ).to(device)
        
        # เลือก optimizer
        if param_dict['optimizer'] == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=param_dict['learning_rate'])
        elif param_dict['optimizer'] == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=param_dict['learning_rate'])
        else:
            optimizer = optim.RMSprop(model.parameters(), lr=param_dict['learning_rate'])
        
        # เลือก loss function
        criterion = nn.CrossEntropyLoss()
        
        # ฝึกสอนและประเมินผล
        results = train_and_evaluate(
            model, criterion, optimizer,
            X_train, y_train, X_val, y_val,
            batch_size=param_dict['batch_size'],
            epochs=param_dict['epochs'],
            device=device,
            early_stopping=True,
            patience=5
        )
        
        # จัดเก็บผลลัพธ์
        experiment_results = {**param_dict, **{
            'val_accuracy': results['best_val_accuracy'],
            'val_f1': results['best_val_f1'],
            'val_loss': results['best_val_loss'],
            'epochs_completed': results['epochs_completed'],
            'training_time': results['training_time']
        }}
        all_results.append(experiment_results)
        
        # ตรวจสอบว่าเป็น best model หรือไม่
        if results['best_val_accuracy'] > best_accuracy:
            best_accuracy = results['best_val_accuracy']
            best_f1 = results['best_val_f1']
            best_params = param_dict
            best_results = results
            
        print(f"การทดลองที่ {i+1}/{len(param_combinations)}:")
        print(f"  Params: {param_dict}")
        print(f"  Accuracy: {results['best_val_accuracy']:.4f}, F1: {results['best_val_f1']:.4f}")
        print(f"  Loss: {results['best_val_loss']:.4f}")
        print(f"  Time: {results['training_time']:.2f} วินาที")
        print("-" * 50)
    
    # แสดงผลลัพธ์ทั้งหมด
    results_df = pd.DataFrame(all_results)
    print("\nผลลัพธ์ทั้งหมด:")
    print(tabulate(results_df, headers='keys', tablefmt='psql'))
    
    # บันทึกผลลัพธ์เป็นไฟล์ CSV
    results_df.to_csv('results/hyperparameter_tuning_results.csv', index=False)
    
    print("\nHyperparameters ที่ดีที่สุด:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")
    print(f"Validation Accuracy: {best_accuracy:.4f}")
    print(f"Validation F1 Score: {best_f1:.4f}")
    
    # สร้างกราฟเปรียบเทียบ
    plot_learning_curves(best_results, best_params)
    plot_hyperparameter_comparison(results_df)
    
    return {'best_params': best_params, 'best_results': best_results, 'all_results': all_results}

def plot_learning_curves(results, params):
    """
    สร้างกราฟแสดง learning curves ของโมเดลที่ดีที่สุด
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot loss
    ax1.plot(results['train_losses'], label='Train Loss')
    ax1.plot(results['val_losses'], label='Validation Loss')
    ax1.set_title('Loss Curves')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracy
    ax2.plot(results['val_accuracies'], label='Validation Accuracy')
    ax2.plot(results['val_f1_scores'], label='Validation F1 Score')
    ax2.set_title('Metrics')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Score')
    ax2.legend()
    ax2.grid(True)
    
    plt.suptitle(f'Learning Curves - Best Model\n(lr={params["learning_rate"]}, batch={params["batch_size"]}, '
                f'hidden={params["hidden_size"]}, optimizer={params["optimizer"]})')
    plt.tight_layout()
    plt.savefig('plots/best_model_learning_curves.png')
    plt.close()

def plot_hyperparameter_comparison(results_df):
    """
    สร้างกราฟเปรียบเทียบผลกระทบของ hyperparameters ต่างๆ
    """
    # 1. เปรียบเทียบ learning rate
    plt.figure(figsize=(12, 8))
    for optimizer in results_df['optimizer'].unique():
        data = results_df[results_df['optimizer'] == optimizer]
        plt.plot(data['learning_rate'], data['val_accuracy'], 'o-', label=f'Optimizer: {optimizer}')
    
    plt.title('ผลกระทบของ Learning Rate ต่อความแม่นยำ')
    plt.xlabel('Learning Rate')
    plt.ylabel('Validation Accuracy')
    plt.xscale('log')
    plt.grid(True)
    plt.legend()
    plt.savefig('plots/learning_rate_comparison.png')
    plt.close()
    
    # 2. เปรียบเทียบ batch size
    plt.figure(figsize=(12, 8))
    for optimizer in results_df['optimizer'].unique():
        data = results_df[results_df['optimizer'] == optimizer]
        plt.plot(data['batch_size'], data['val_accuracy'], 'o-', label=f'Optimizer: {optimizer}')
    
    plt.title('ผลกระทบของ Batch Size ต่อความแม่นยำ')
    plt.xlabel('Batch Size')
    plt.ylabel('Validation Accuracy')
    plt.grid(True)
    plt.legend()
    plt.savefig('plots/batch_size_comparison.png')
    plt.close()
    
    # 3. เปรียบเทียบ hidden size
    plt.figure(figsize=(12, 8))
    for optimizer in results_df['optimizer'].unique():
        data = results_df[results_df['optimizer'] == optimizer]
        plt.plot(data['hidden_size'], data['val_accuracy'], 'o-', label=f'Optimizer: {optimizer}')
    
    plt.title('ผลกระทบของ Hidden Size ต่อความแม่นยำ')
    plt.xlabel('Hidden Size')
    plt.ylabel('Validation Accuracy')
    plt.grid(True)
    plt.legend()
    plt.savefig('plots/hidden_size_comparison.png')
    plt.close()
    
    # 4. เปรียบเทียบ training time
    plt.figure(figsize=(12, 8))
    for optimizer in results_df['optimizer'].unique():
        data = results_df[results_df['optimizer'] == optimizer]
        plt.scatter(data['training_time'], data['val_accuracy'], label=f'Optimizer: {optimizer}')
    
    plt.title('ความสัมพันธ์ระหว่างเวลาฝึกสอนและความแม่นยำ')
    plt.xlabel('Training Time (seconds)')
    plt.ylabel('Validation Accuracy')
    plt.grid(True)
    plt.legend()
    plt.savefig('plots/training_time_comparison.png')
    plt.close()
    
    # 5. สร้าง heatmap สำหรับ optimizer, learning rate และ batch size
    plt.figure(figsize=(15, 10))
    
    for i, optimizer in enumerate(results_df['optimizer'].unique()):
        plt.subplot(1, len(results_df['optimizer'].unique()), i+1)
        
        # กรองข้อมูลเฉพาะ optimizer นี้
        optimizer_data = results_df[results_df['optimizer'] == optimizer].copy()
        
        # เตรียมข้อมูลสำหรับ heatmap
        pivot_data = optimizer_data.pivot_table(
            values='val_accuracy', 
            index='learning_rate', 
            columns='batch_size', 
            aggfunc='mean'
        )
        
        # สร้าง heatmap
        im = plt.imshow(pivot_data, cmap='viridis', aspect='auto')
        plt.colorbar(im, label='Validation Accuracy')
        
        # ปรับแต่ง tick labels
        plt.xticks(range(len(pivot_data.columns)), pivot_data.columns)
        plt.yticks(range(len(pivot_data.index)), [f"{x:.5f}" for x in pivot_data.index])
        
        plt.title(f'Optimizer: {optimizer}')
        plt.xlabel('Batch Size')
        plt.ylabel('Learning Rate')
    
    plt.suptitle('ความแม่นยำตาม Optimizer, Learning Rate และ Batch Size', fontsize=16)
    plt.tight_layout()
    plt.savefig('plots/hyperparameter_heatmap.png')
    plt.close()

def manual_tuning_example():
    """
    ตัวอย่างการปรับ hyperparameters แบบ manual
    """
    # สร้างข้อมูล
    X, y = generate_data(n_samples=2000, n_features=20, n_classes=2)
    
    # แบ่งข้อมูลเป็น training และ validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # กำหนด hyperparameters ที่จะลองปรับ
    param_grid = {
        'learning_rate': [0.001, 0.01, 0.1],
        'batch_size': [16, 32, 64],
        'hidden_size': [32, 64, 128],
        'optimizer': ['adam', 'sgd', 'rmsprop'],
        'epochs': [50]  # กำหนดค่าคงที่ (ใช้ early stopping)
    }
    
    # เลือกอุปกรณ์การคำนวณ (GPU ถ้ามี)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"ใช้อุปกรณ์: {device}")
    
    # ทำการค้นหา hyperparameters แบบ manual
    tuning_results = grid_search_manual(X_train, y_train, X_val, y_val, param_grid, device)
    
    # นำ hyperparameters ที่ดีที่สุดไปเทรนโมเดลสุดท้าย
    best_params = tuning_results['best_params']
    print("\nเทรนโมเดลสุดท้ายด้วย hyperparameters ที่ดีที่สุด:")
    
    # สร้างโมเดลด้วย hyperparameters ที่ดีที่สุด
    final_model = SimpleClassifier(
        input_size=X_train.shape[1], 
        hidden_size=best_params['hidden_size'], 
        num_classes=len(np.unique(y_train))
    ).to(device)
    
    # เลือก optimizer
    if best_params['optimizer'] == 'adam':
        optimizer = optim.Adam(final_model.parameters(), lr=best_params['learning_rate'])
    elif best_params['optimizer'] == 'sgd':
        optimizer = optim.SGD(final_model.parameters(), lr=best_params['learning_rate'])
    else:
        optimizer = optim.RMSprop(final_model.parameters(), lr=best_params['learning_rate'])
    
    # ฝึกสอนโมเดลสุดท้าย
    criterion = nn.CrossEntropyLoss()
    final_results = train_and_evaluate(
        final_model, criterion, optimizer,
        X_train, y_train, X_val, y_val,
        batch_size=best_params['batch_size'],
        epochs=100,  # เพิ่ม epochs สำหรับโมเดลสุดท้าย
        device=device,
        early_stopping=True,
        patience=10  # เพิ่ม patience สำหรับโมเดลสุดท้าย
    )
    
    print("\nผลลัพธ์สุดท้าย:")
    print(f"Validation Accuracy: {final_results['best_val_accuracy']:.4f}")
    print(f"Validation F1 Score: {final_results['best_val_f1']:.4f}")
    print(f"Training completed in {final_results['epochs_completed']} epochs")
    print(f"Training time: {final_results['training_time']:.2f} seconds")
    
    # สร้างกราฟสำหรับโมเดลสุดท้าย
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.plot(final_results['train_losses'], label='Train Loss')
    plt.plot(final_results['val_losses'], label='Validation Loss')
    plt.title('Loss Curves - Final Model')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(final_results['val_accuracies'], label='Validation Accuracy')
    plt.plot(final_results['val_f1_scores'], label='Validation F1 Score')
    plt.title('Metrics - Final Model')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    
    plt.suptitle(f'Final Model with Best Hyperparameters\n'
                f'(lr={best_params["learning_rate"]}, batch={best_params["batch_size"]}, '
                f'hidden={best_params["hidden_size"]}, optimizer={best_params["optimizer"]})')
    plt.tight_layout()
    plt.savefig('plots/final_model_results.png')
    plt.close()

    # สรุปผลการปรับ hyperparameters
    print("\nสรุปการปรับ Hyperparameters:")
    print(f"1. เราลอง {len(tuning_results['all_results'])} ชุด hyperparameters ต่างกัน")
    print(f"2. Hyperparameters ที่ดีที่สุด:")
    for param, value in best_params.items():
        print(f"   - {param}: {value}")
    print(f"3. ความแม่นยำที่ดีที่สุด: {final_results['best_val_accuracy']:.4f}")
    print(f"4. F1 Score ที่ดีที่สุด: {final_results['best_val_f1']:.4f}")
    
    # แสดงข้อสังเกตจากการปรับ hyperparameters
    results_df = pd.DataFrame(tuning_results['all_results'])
    
    print("\nข้อสังเกตจากการปรับ Hyperparameters:")
    
    # ผลกระทบของ learning rate
    lr_effect = results_df.groupby('learning_rate')['val_accuracy'].mean()
    best_lr = lr_effect.idxmax()
    print(f"1. Learning Rate: ค่าที่ให้ความแม่นยำเฉลี่ยสูงสุดคือ {best_lr}")
    
    # ผลกระทบของ batch size
    batch_effect = results_df.groupby('batch_size')['val_accuracy'].mean()
    best_batch = batch_effect.idxmax()
    print(f"2. Batch Size: ค่าที่ให้ความแม่นยำเฉลี่ยสูงสุดคือ {best_batch}")
    
    # ผลกระทบของ hidden size
    hidden_effect = results_df.groupby('hidden_size')['val_accuracy'].mean()
    best_hidden = hidden_effect.idxmax()
    print(f"3. Hidden Size: ค่าที่ให้ความแม่นยำเฉลี่ยสูงสุดคือ {best_hidden}")
    
    # ผลกระทบของ optimizer
    optimizer_effect = results_df.groupby('optimizer')['val_accuracy'].mean()
    best_optimizer = optimizer_effect.idxmax()
    print(f"4. Optimizer: ตัวที่ให้ความแม่นยำเฉลี่ยสูงสุดคือ {best_optimizer}")
    
    # ความเร็วในการเทรน
    optimizer_time = results_df.groupby('optimizer')['training_time'].mean()
    fastest_optimizer = optimizer_time.idxmin()
    print(f"5. เวลาเทรนเฉลี่ยตาม Optimizer:")
    for opt, time_val in optimizer_time.items():
        print(f"   - {opt}: {time_val:.2f} วินาที")

if __name__ == "__main__":
    manual_tuning_example() 