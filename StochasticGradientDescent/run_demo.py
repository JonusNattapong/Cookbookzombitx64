import numpy as np
import matplotlib.pyplot as plt
import os
import time
from stochastic_gradient_descent import StochasticGradientDescent
import sys

# เพิ่มเส้นทางไปยัง DatasetGenerator เพื่อใช้งาน dataset_generator
sys.path.append('../DatasetGenerator')
from DatasetGenerator.dataset_generator import generate_linear_dataset, visualize_dataset, save_dataset

def run_demo(n_samples=100, n_features=1, noise=15.0, learning_rate=0.1, epochs=50, batch_size=10):
    """
    สาธิตการทำงานของ Stochastic Gradient Descent
    """
    print("=" * 50)
    print("     การสาธิต Stochastic Gradient Descent     ")
    print("=" * 50)
    
    # สร้างโฟลเดอร์สำหรับเก็บข้อมูลและรูปภาพ
    for folder in ['./data', './plots']:
        if not os.path.exists(folder):
            os.makedirs(folder)
    
    # สร้างชุดข้อมูล
    print("\n1. กำลังสร้างชุดข้อมูล...")
    X, y = generate_linear_dataset(n_samples=n_samples, n_features=n_features, noise=noise, dataset_name="sgd_dataset")
    print(f"   จำนวนตัวอย่าง: {n_samples}")
    print(f"   จำนวนคุณลักษณะ: {n_features}")
    print(f"   ระดับความรบกวน: {noise}")
    
    # แสดงภาพชุดข้อมูล
    print("\n2. การแสดงภาพชุดข้อมูล...")
    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, alpha=0.7)
    plt.title("ชุดข้อมูลสำหรับการถดถอยเชิงเส้น (SGD)")
    plt.xlabel("X")
    plt.ylabel("y")
    plt.grid(True, alpha=0.3)
    plt.savefig('./plots/dataset_visualization.png')
    plt.close()
    print("บันทึกภาพไปที่ './plots/dataset_visualization.png'")
    
    # บันทึกชุดข้อมูล
    print("\n3. กำลังบันทึกชุดข้อมูล...")
    save_dataset(X, y, None, None, dataset_name="sgd_dataset")
    
    # สร้างและฝึกโมเดล
    print("\n4. กำลังฝึกโมเดลด้วย Stochastic Gradient Descent...")
    start_time = time.time()
    model = StochasticGradientDescent(learning_rate=learning_rate, epochs=epochs, batch_size=batch_size)
    model.fit(X, y)
    training_time = time.time() - start_time
    
    # แสดงผลลัพธ์
    print("\n5. ผลลัพธ์ของโมเดล:")
    y_pred = model.predict(X)
    mse = model.compute_loss(y, y_pred)
    print(f"   ค่าน้ำหนัก (Weights): {model.weights}")
    print(f"   ค่า Bias: {model.bias}")
    print(f"   ค่าความสูญเสีย (MSE): {mse:.6f}")
    print(f"   เวลาที่ใช้ในการฝึกโมเดล: {training_time:.4f} วินาที")
    
    # แสดงกราฟ
    print("\n6. กำลังสร้างกราฟ...")
    model.plot_loss_history()
    model.plot_regression_line(X, y)
    
    print("\nการสาธิตเสร็จสิ้น!")
    print("=" * 50)
    
    return model

def compare_batch_sizes(n_samples=100, n_features=1, noise=15.0, learning_rate=0.1, epochs=50):
    """
    เปรียบเทียบขนาด batch size ต่างๆ
    """
    print("=" * 50)
    print("     เปรียบเทียบขนาด Batch Size ต่างๆ     ")
    print("=" * 50)
    
    # สร้างชุดข้อมูล
    X, y = generate_linear_dataset(n_samples=n_samples, n_features=n_features, noise=noise, dataset_name="sgd_comparison_dataset")
    
    # ขนาด batch size ที่ต้องการทดสอบ
    batch_sizes = [1, 5, 10, 20, n_samples]  # n_samples คือ full batch (Vanilla GD)
    
    results = []
    
    plt.figure(figsize=(12, 8))
    
    for i, batch_size in enumerate(batch_sizes):
        print(f"\nกำลังทดสอบ batch_size = {batch_size}...")
        
        # สร้างและฝึกโมเดล
        start_time = time.time()
        model = StochasticGradientDescent(learning_rate=learning_rate, epochs=epochs, batch_size=batch_size)
        model.fit(X, y)
        training_time = time.time() - start_time
        
        # ทำนายและคำนวณค่าความสูญเสีย
        y_pred = model.predict(X)
        mse = model.compute_loss(y, y_pred)
        
        # เก็บผลลัพธ์
        results.append({
            'batch_size': batch_size,
            'weights': model.weights,
            'bias': model.bias,
            'mse': mse,
            'training_time': training_time,
            'epoch_loss_history': model.epoch_loss_history
        })
        
        # พล็อตประวัติค่าความสูญเสีย
        label = f"Batch Size = {batch_size}"
        if batch_size == n_samples:
            label = "Full Batch (Vanilla GD)"
        plt.plot(model.epoch_loss_history, label=label)
    
    plt.title("เปรียบเทียบการลู่เข้าของค่าความสูญเสียสำหรับขนาด Batch Size ต่างๆ")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('./plots/batch_size_comparison.png')
    plt.close()
    
    # แสดงผลลัพธ์
    print("\nผลการเปรียบเทียบ:")
    for result in results:
        print(f"\nBatch Size = {result['batch_size']}:")
        print(f"  MSE: {result['mse']:.6f}")
        print(f"  เวลาที่ใช้: {result['training_time']:.4f} วินาที")
    
    return results

if __name__ == "__main__":
    # 1. ทดสอบการทำงานของ Stochastic Gradient Descent
    model = run_demo(n_samples=100, 
                    n_features=1, 
                    noise=15.0, 
                    learning_rate=0.1, 
                    epochs=50, 
                    batch_size=10)
    
    # 2. เปรียบเทียบขนาด batch size ต่างๆ
    results = compare_batch_sizes(n_samples=100, 
                                 n_features=1, 
                                 noise=15.0, 
                                 learning_rate=0.1, 
                                 epochs=50) 