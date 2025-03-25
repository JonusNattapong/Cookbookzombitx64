import numpy as np
import matplotlib.pyplot as plt
import os
from DatasetGenerator.dataset_generator import load_dataset, generate_linear_dataset

class VanillaGradientDescent:
    """
    คลาสสำหรับการทำ Vanilla Gradient Descent
    """
    def __init__(self, learning_rate=0.01, max_iterations=1000, tolerance=1e-6):
        """
        Parameters:
        -----------
        learning_rate : float
            อัตราการเรียนรู้ (learning rate)
        max_iterations : int
            จำนวนรอบสูงสุดในการเรียนรู้
        tolerance : float
            ค่าความคลาดเคลื่อนที่ยอมรับได้สำหรับการหยุดการเรียนรู้
        """
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.weights = None
        self.bias = None
        self.loss_history = []
    
    def initialize_parameters(self, n_features):
        """
        กำหนดค่าเริ่มต้นของพารามิเตอร์
        """
        self.weights = np.zeros(n_features)
        self.bias = 0
    
    def compute_predictions(self, X):
        """
        คำนวณค่าทำนาย
        """
        return np.dot(X, self.weights) + self.bias
    
    def compute_loss(self, y_true, y_pred):
        """
        คำนวณค่าความสูญเสีย (Mean Squared Error)
        """
        m = len(y_true)
        loss = np.sum((y_pred - y_true) ** 2) / (2 * m)
        return loss
    
    def compute_gradients(self, X, y_true, y_pred):
        """
        คำนวณค่า gradient ของ loss function
        """
        m = len(y_true)
        dw = np.dot(X.T, (y_pred - y_true)) / m
        db = np.sum(y_pred - y_true) / m
        return dw, db
    
    def update_parameters(self, dw, db):
        """
        อัปเดตค่าพารามิเตอร์
        """
        self.weights -= self.learning_rate * dw
        self.bias -= self.learning_rate * db
    
    def fit(self, X, y):
        """
        ฝึกโมเดลด้วย Vanilla Gradient Descent
        """
        # กำหนดค่าเริ่มต้น
        n_samples, n_features = X.shape
        self.initialize_parameters(n_features)
        
        # วนรอบการเรียนรู้
        for i in range(self.max_iterations):
            # คำนวณค่าทำนาย
            y_pred = self.compute_predictions(X)
            
            # คำนวณค่าความสูญเสีย
            loss = self.compute_loss(y, y_pred)
            self.loss_history.append(loss)
            
            # คำนวณ gradient
            dw, db = self.compute_gradients(X, y, y_pred)
            
            # อัปเดตพารามิเตอร์
            self.update_parameters(dw, db)
            
            # ตรวจสอบการลู่เข้า
            if i > 0 and abs(self.loss_history[i] - self.loss_history[i-1]) < self.tolerance:
                print(f"สิ้นสุดการเรียนรู้ที่รอบที่ {i} เนื่องจาก loss ลู่เข้าแล้ว")
                break
                
            # แสดงความคืบหน้า
            if (i + 1) % 100 == 0:
                print(f"รอบที่ {i+1}/{self.max_iterations}, Loss: {loss:.6f}")
        
        return self
    
    def predict(self, X):
        """
        ทำนายผลลัพธ์
        """
        return self.compute_predictions(X)
    
    def plot_loss_history(self):
        """
        แสดงกราฟประวัติค่าความสูญเสีย
        """
        plt.figure(figsize=(10, 6))
        plt.plot(self.loss_history)
        plt.title("ประวัติค่าความสูญเสีย")
        plt.xlabel("รอบการเรียนรู้")
        plt.ylabel("ค่าความสูญเสีย (MSE)")
        plt.grid(True, alpha=0.3)
        
        # สร้างโฟลเดอร์เพื่อบันทึกรูปภาพ
        if not os.path.exists('./plots'):
            os.makedirs('./plots')
            
        plt.savefig('./plots/loss_history.png')
        plt.close()
        
        print("บันทึกกราฟประวัติค่าความสูญเสียไปที่ './plots/loss_history.png'")
    
    def plot_regression_line(self, X, y):
        """
        แสดงเส้นถดถอย
        """
        plt.figure(figsize=(10, 6))
        plt.scatter(X, y, alpha=0.7)
        
        # สร้างข้อมูลสำหรับการพล็อตเส้นถดถอย
        x_min, x_max = X.min(), X.max()
        x_line = np.linspace(x_min, x_max, 100).reshape(-1, 1)
        y_line = self.predict(x_line)
        
        plt.plot(x_line, y_line, color='r', label=f'y = {self.weights[0]:.4f}x + {self.bias:.4f}')
        plt.title("เส้นถดถอยเชิงเส้น")
        plt.xlabel("X")
        plt.ylabel("y")
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # สร้างโฟลเดอร์เพื่อบันทึกรูปภาพ
        if not os.path.exists('./plots'):
            os.makedirs('./plots')
            
        plt.savefig('./plots/regression_line.png')
        plt.close()
        
        print("บันทึกกราฟเส้นถดถอยไปที่ './plots/regression_line.png'")

if __name__ == "__main__":
    # โหลดหรือสร้างชุดข้อมูล
    try:
        X, y = load_dataset()
        print("โหลดชุดข้อมูลสำเร็จ")
    except:
        print("ไม่พบชุดข้อมูล กำลังสร้างชุดข้อมูลใหม่...")
        X, y = generate_linear_dataset(n_samples=100, n_features=1, noise=15.0)
    
    # สร้างและฝึกโมเดล
    model = VanillaGradientDescent(learning_rate=0.1, max_iterations=1000)
    model.fit(X, y)
    
    # แสดงผลลัพธ์
    y_pred = model.predict(X)
    mse = model.compute_loss(y, y_pred)
    print(f"ค่าน้ำหนัก (Weights): {model.weights}")
    print(f"ค่า Bias: {model.bias}")
    print(f"ค่าความสูญเสีย (MSE): {mse:.6f}")
    
    # แสดงกราฟ
    model.plot_loss_history()
    model.plot_regression_line(X, y) 