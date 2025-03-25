import numpy as np
import matplotlib.pyplot as plt
import os
import time
from DatasetGenerator.dataset_generator import load_dataset, generate_linear_dataset

class StochasticGradientDescent:
    """
    คลาสสำหรับการทำ Stochastic Gradient Descent (SGD)
    
    แนวคิด: อัปเดตน้ำหนักโดยใช้ gradient จากข้อมูลตัวอย่างเดียว (หรือ batch เล็กๆ) แทนทั้งชุด
    วิธีการ:
    - สุ่มเลือกข้อมูล 1 ตัวอย่างหรือ batch
    - คำนวณ gradient และอัปเดตน้ำหนักทันที
    """
    def __init__(self, learning_rate=0.01, epochs=100, batch_size=1, tolerance=1e-6):
        """
        Parameters:
        -----------
        learning_rate : float
            อัตราการเรียนรู้ (learning rate)
        epochs : int
            จำนวนรอบการเรียนรู้ทั้งหมด (epochs)
        batch_size : int
            ขนาด batch (จำนวนตัวอย่างที่ใช้ในการคำนวณ gradient แต่ละครั้ง)
            ถ้า batch_size=1 คือ pure stochastic gradient descent
            ถ้า batch_size>1 คือ mini-batch stochastic gradient descent
        tolerance : float
            ค่าความคลาดเคลื่อนที่ยอมรับได้สำหรับการหยุดการเรียนรู้
        """
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.tolerance = tolerance
        self.weights = None
        self.bias = None
        self.loss_history = []
        self.epoch_loss_history = []
    
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
    
    def create_batches(self, X, y):
        """
        สร้าง batches จากข้อมูล
        """
        n_samples = len(y)
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        
        # คำนวณจำนวน batches
        n_batches = int(np.ceil(n_samples / self.batch_size))
        
        batches = []
        for i in range(n_batches):
            # ดึงดัชนีสำหรับ batch นี้
            batch_indices = indices[i * self.batch_size:(i + 1) * self.batch_size]
            batch_X = X[batch_indices]
            batch_y = y[batch_indices]
            batches.append((batch_X, batch_y))
        
        return batches
    
    def fit(self, X, y):
        """
        ฝึกโมเดลด้วย Stochastic Gradient Descent
        """
        # กำหนดค่าเริ่มต้น
        n_samples, n_features = X.shape
        self.initialize_parameters(n_features)
        
        # วนรอบการเรียนรู้ (epochs)
        for epoch in range(self.epochs):
            epoch_start_time = time.time()
            
            # สร้าง batches
            batches = self.create_batches(X, y)
            
            # เก็บค่า loss ของแต่ละ batch ในรอบนี้
            batch_losses = []
            
            # วนลูปสำหรับแต่ละ batch
            for batch_X, batch_y in batches:
                # คำนวณค่าทำนาย
                batch_y_pred = self.compute_predictions(batch_X)
                
                # คำนวณ gradient
                dw, db = self.compute_gradients(batch_X, batch_y, batch_y_pred)
                
                # อัปเดตพารามิเตอร์
                self.update_parameters(dw, db)
                
                # คำนวณค่าความสูญเสียสำหรับ batch นี้
                batch_loss = self.compute_loss(batch_y, batch_y_pred)
                batch_losses.append(batch_loss)
                self.loss_history.append(batch_loss)
            
            # คำนวณค่าความสูญเสียเฉลี่ยของรอบนี้
            epoch_loss = np.mean(batch_losses)
            self.epoch_loss_history.append(epoch_loss)
            
            # ตรวจสอบการลู่เข้า
            if epoch > 0 and abs(self.epoch_loss_history[epoch] - self.epoch_loss_history[epoch-1]) < self.tolerance:
                print(f"สิ้นสุดการเรียนรู้ที่รอบที่ {epoch+1} เนื่องจาก loss ลู่เข้าแล้ว")
                break
                
            # แสดงความคืบหน้า
            epoch_time = time.time() - epoch_start_time
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"รอบที่ {epoch+1}/{self.epochs}, Loss: {epoch_loss:.6f}, เวลา: {epoch_time:.4f} วินาที")
        
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
        plt.figure(figsize=(12, 5))
        
        # พล็อตค่า loss ของแต่ละ batch
        plt.subplot(1, 2, 1)
        plt.plot(self.loss_history)
        plt.title("ประวัติค่าความสูญเสียของแต่ละ batch")
        plt.xlabel("Batch")
        plt.ylabel("ค่าความสูญเสีย (MSE)")
        plt.grid(True, alpha=0.3)
        
        # พล็อตค่า loss เฉลี่ยของแต่ละ epoch
        plt.subplot(1, 2, 2)
        plt.plot(self.epoch_loss_history)
        plt.title("ประวัติค่าความสูญเสียเฉลี่ยของแต่ละรอบ")
        plt.xlabel("Epoch")
        plt.ylabel("ค่าความสูญเสียเฉลี่ย (MSE)")
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # สร้างโฟลเดอร์เพื่อบันทึกรูปภาพ
        if not os.path.exists('./plots'):
            os.makedirs('./plots')
            
        plt.savefig('./plots/sgd_loss_history.png')
        plt.close()
        
        print("บันทึกกราฟประวัติค่าความสูญเสียไปที่ './plots/sgd_loss_history.png'")
    
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
        plt.title("เส้นถดถอยเชิงเส้น (SGD)")
        plt.xlabel("X")
        plt.ylabel("y")
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # สร้างโฟลเดอร์เพื่อบันทึกรูปภาพ
        if not os.path.exists('./plots'):
            os.makedirs('./plots')
            
        plt.savefig('./plots/sgd_regression_line.png')
        plt.close()
        
        print("บันทึกกราฟเส้นถดถอยไปที่ './plots/sgd_regression_line.png'")

if __name__ == "__main__":
    # โหลดหรือสร้างชุดข้อมูล
    try:
        X, y = load_dataset()
        print("โหลดชุดข้อมูลสำเร็จ")
    except:
        print("ไม่พบชุดข้อมูล กำลังสร้างชุดข้อมูลใหม่...")
        X, y = generate_linear_dataset(n_samples=100, n_features=1, noise=15.0)
    
    # สร้างและฝึกโมเดล
    model = StochasticGradientDescent(learning_rate=0.1, epochs=50, batch_size=10)
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