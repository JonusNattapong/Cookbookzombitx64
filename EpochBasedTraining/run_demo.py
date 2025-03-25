import numpy as np
import matplotlib.pyplot as plt
import os
import time
import sys
from epoch_based_training import EpochBasedTraining

# เพิ่มเส้นทางไปยัง DatasetGenerator เพื่อใช้งาน dataset_generator
sys.path.append('../DatasetGenerator')
try:
    from DatasetGenerator.dataset_generator import generate_linear_dataset, visualize_dataset, save_dataset
except ImportError:
    print("ไม่สามารถนำเข้าโมดูล dataset_generator ได้")

def run_demo(n_samples=100, n_features=1, noise=15.0, learning_rate=0.1, epochs=50):
    """
    สาธิตการทำงานของ Epoch-Based Training
    """
    print("=" * 50)
    print("     การสาธิต Epoch-Based Training     ")
    print("=" * 50)
    
    # สร้างโฟลเดอร์สำหรับเก็บข้อมูลและรูปภาพ
    for folder in ['./data', './plots']:
        if not os.path.exists(folder):
            os.makedirs(folder)
    
    # สร้างชุดข้อมูล
    print("\n1. กำลังสร้างชุดข้อมูล...")
    X, y = generate_linear_dataset(n_samples=n_samples, n_features=n_features, noise=noise, dataset_name="epoch_based_dataset")
    print(f"   จำนวนตัวอย่าง: {n_samples}")
    print(f"   จำนวนคุณลักษณะ: {n_features}")
    print(f"   ระดับความรบกวน: {noise}")
    
    # แสดงภาพชุดข้อมูล
    print("\n2. การแสดงภาพชุดข้อมูล...")
    visualize_dataset(X, y, dataset_name="epoch_based_dataset")
    
    # บันทึกชุดข้อมูล
    print("\n3. กำลังบันทึกชุดข้อมูล...")
    save_dataset(X, y, None, None, dataset_name="epoch_based_dataset")
    
    # สร้างและฝึกโมเดล
    print(f"\n4. กำลังฝึกโมเดลด้วย Epoch-Based Training...")
    start_time = time.time()
    model = EpochBasedTraining(learning_rate=learning_rate, epochs=epochs)
    model.fit(X, y)
    training_time = time.time() - start_time
    
    # แสดงผลลัพธ์
    print("\n5. ผลลัพธ์ของโมเดล:")
    y_pred = model.predict(X)
    mse = model.compute_loss(y, y_pred)
    print(f"   ค่าน้ำหนัก (Weights): {model.weights.detach().numpy()}")
    print(f"   ค่า Bias: {model.bias.detach().numpy()}")
    print(f"   ค่าความสูญเสีย (MSE): {mse:.6f}")
    print(f"   เวลาที่ใช้ในการฝึกโมเดล: {training_time:.4f} วินาที")
    
    # แสดงกราฟ
    print("\n6. กำลังสร้างกราฟ...")
    model.plot_loss_history()
    model.plot_regression_line(X, y)
    
    print("\nการสาธิตเสร็จสิ้น!")
    print("=" * 50)
    
    return model

def compare_epochs(n_samples=100, n_features=1, noise=15.0, learning_rate=0.1):
    """
    เปรียบเทียบจำนวน epochs ต่างๆ
    """
    print("=" * 50)
    print("     เปรียบเทียบจำนวน Epochs ต่างๆ     ")
    print("=" * 50)
    
    # สร้างชุดข้อมูล
    X, y = generate_linear_dataset(n_samples=n_samples, n_features=n_features, noise=noise, dataset_name="epoch_comparison_dataset")
    
    # จำนวน epochs ที่ต้องการทดสอบ
    epochs_list = [10, 25, 50, 100, 200]
    
    results = []
    
    plt.figure(figsize=(12, 8))
    
    for epochs in epochs_list:
        print(f"\nกำลังทดสอบ epochs = {epochs}...")
        
        # สร้างและฝึกโมเดล
        start_time = time.time()
        model = EpochBasedTraining(learning_rate=learning_rate, epochs=epochs)
        model.fit(X, y)
        training_time = time.time() - start_time
        
        # ทำนายและคำนวณค่าความสูญเสีย
        y_pred = model.predict(X)
        mse = model.compute_loss(y, y_pred)
        
        # เก็บผลลัพธ์
        results.append({
            'epochs': epochs,
            'weights': model.weights.detach().numpy(),
            'bias': model.bias.detach().numpy(),
            'mse': mse,
            'training_time': training_time,
            'epoch_loss_history': model.epoch_loss_history
        })
        
        # พล็อตประวัติค่าความสูญเสีย
        plt.plot(model.epoch_loss_history, label=f"Epochs = {epochs}")
    
    plt.title("เปรียบเทียบการลู่เข้าของค่าความสูญเสียสำหรับจำนวน Epochs ต่างๆ")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('./plots/epoch_comparison.png')
    plt.close()
    
    # แสดงผลลัพธ์
    print("\nผลการเปรียบเทียบ:")
    for result in results:
        print(f"\nEpochs = {result['epochs']}:")
        print(f"  MSE: {result['mse']:.6f}")
        print(f"  เวลาที่ใช้: {result['training_time']:.4f} วินาที")
    
    return results

if __name__ == "__main__":
    # 1. ทดสอบการทำงานของ Epoch-Based Training
    model = run_demo(n_samples=100, 
                    n_features=1, 
                    noise=15.0, 
                    learning_rate=0.1, 
                    epochs=50)
    
    # 2. เปรียบเทียบจำนวน epochs ต่างๆ
    results = compare_epochs(n_samples=100, 
                         n_features=1, 
                         noise=15.0, 
                         learning_rate=0.1) 