import numpy as np
import matplotlib.pyplot as plt
from DatasetGenerator.dataset_generator import generate_linear_dataset, visualize_dataset, save_dataset
from vanilla_gradient_descent import VanillaGradientDescent
import time

def run_demo(n_samples=100, n_features=1, noise=15.0, learning_rate=0.1, max_iterations=1000):
    """
    สาธิตการทำงานของ Vanilla Gradient Descent
    """
    print("=" * 50)
    print("     การสาธิต Vanilla Gradient Descent     ")
    print("=" * 50)
    
    # สร้างชุดข้อมูล
    print("\n1. กำลังสร้างชุดข้อมูล...")
    X, y = generate_linear_dataset(n_samples=n_samples, n_features=n_features, noise=noise, dataset_name="vanilla_gd_dataset")
    print(f"   จำนวนตัวอย่าง: {n_samples}")
    print(f"   จำนวนคุณลักษณะ: {n_features}")
    print(f"   ระดับความรบกวน: {noise}")
    
    # แสดงภาพชุดข้อมูล
    print("\n2. การแสดงภาพชุดข้อมูล...")
    visualize_dataset(X, y, dataset_name="vanilla_gd_dataset")
    
    # บันทึกชุดข้อมูล
    print("\n3. กำลังบันทึกชุดข้อมูล...")
    save_dataset(X, y, None, None, dataset_name="vanilla_gd_dataset")
    
    # สร้างและฝึกโมเดล
    print("\n4. กำลังฝึกโมเดลด้วย Vanilla Gradient Descent...")
    start_time = time.time()
    model = VanillaGradientDescent(learning_rate=learning_rate, max_iterations=max_iterations)
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

if __name__ == "__main__":
    # รันการสาธิต
    model = run_demo(n_samples=100, 
                    n_features=1, 
                    noise=15.0, 
                    learning_rate=0.1, 
                    max_iterations=1000) 