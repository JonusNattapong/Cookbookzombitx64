import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import time
from sklearn.metrics import mean_squared_error
import os

class EpochBasedTraining:
    def __init__(self, learning_rate=0.1, epochs=50):
        """
        เริ่มต้นการเทรนแบบ Epoch-Based
        
        Parameters:
        -----------
        learning_rate : float
            อัตราการเรียนรู้
        epochs : int
            จำนวนรอบในการเทรน
        """
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None
        self.epoch_loss_history = []
        self.training_time = 0
        
    def fit(self, X, y):
        """
        เทรนโมเดลโดยใช้ข้อมูลทั้งหมดในแต่ละ epoch
        
        Parameters:
        -----------
        X : numpy array
            ข้อมูล input
        y : numpy array
            ข้อมูล target
        """
        # แปลงข้อมูลเป็น PyTorch tensors
        X = torch.FloatTensor(X)
        y = torch.FloatTensor(y)
        
        # เริ่มจับเวลา
        start_time = time.time()
        
        # เริ่มต้นพารามิเตอร์
        n_features = X.shape[1]
        self.weights = torch.zeros(n_features, requires_grad=True)
        self.bias = torch.zeros(1, requires_grad=True)
        
        # เก็บประวัติค่าความสูญเสีย
        self.epoch_loss_history = []
        
        print(f"\nเริ่มการเทรน {self.epochs} epochs...")
        print("=" * 50)
        
        for epoch in range(self.epochs):
            # คำนวณการทำนาย
            y_pred = torch.matmul(X, self.weights) + self.bias
            
            # คำนวณค่าความสูญเสีย (MSE)
            loss = torch.mean((y_pred - y) ** 2)
            
            # คำนวณ gradient
            loss.backward()
            
            # อัพเดทพารามิเตอร์
            with torch.no_grad():
                self.weights -= self.learning_rate * self.weights.grad
                self.bias -= self.learning_rate * self.bias.grad
                
                # รีเซ็ต gradient
                self.weights.grad.zero_()
                self.bias.grad.zero_()
            
            # เก็บค่าความสูญเสีย
            self.epoch_loss_history.append(loss.item())
            
            # แสดงความคืบหน้า
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {loss.item():.6f}")
        
        # จบการจับเวลา
        self.training_time = time.time() - start_time
        
        print("\nการเทรนเสร็จสิ้น!")
        print(f"เวลาที่ใช้: {self.training_time:.4f} วินาที")
        print(f"ค่าความสูญเสียสุดท้าย: {self.epoch_loss_history[-1]:.6f}")
        print("=" * 50)
    
    def predict(self, X):
        """
        ทำนายค่า y จากข้อมูล input
        
        Parameters:
        -----------
        X : numpy array
            ข้อมูล input
            
        Returns:
        --------
        y_pred : numpy array
            ค่าที่ทำนายได้
        """
        X = torch.FloatTensor(X)
        y_pred = torch.matmul(X, self.weights) + self.bias
        return y_pred.detach().numpy()
    
    def compute_loss(self, y_true, y_pred):
        """
        คำนวณค่าความสูญเสีย (MSE)
        
        Parameters:
        -----------
        y_true : numpy array
            ค่าจริง
        y_pred : numpy array
            ค่าที่ทำนายได้
            
        Returns:
        --------
        mse : float
            ค่าความสูญเสีย
        """
        return mean_squared_error(y_true, y_pred)
    
    def plot_loss_history(self):
        """
        แสดงกราฟประวัติค่าความสูญเสีย
        """
        plt.figure(figsize=(10, 6))
        plt.plot(self.epoch_loss_history)
        plt.title("ประวัติค่าความสูญเสียระหว่างการเทรน")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True, alpha=0.3)
        
        # สร้างโฟลเดอร์ plots ถ้ายังไม่มี
        if not os.path.exists('./plots'):
            os.makedirs('./plots')
            
        plt.savefig('./plots/epoch_based_loss_history.png')
        plt.close()
        print("บันทึกกราฟประวัติค่าความสูญเสียไปที่ './plots/epoch_based_loss_history.png'")
    
    def plot_regression_line(self, X, y):
        """
        แสดงภาพข้อมูลและเส้นถดถอย
        
        Parameters:
        -----------
        X : numpy array
            ข้อมูล input
        y : numpy array
            ข้อมูล target
        """
        plt.figure(figsize=(10, 6))
        plt.scatter(X, y, alpha=0.7, label='ข้อมูล')
        
        # สร้างจุดสำหรับพล็อตเส้นถดถอย
        X_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
        y_line = self.predict(X_line)
        plt.plot(X_line, y_line, 'r-', label='เส้นถดถอย')
        
        plt.title("ผลการถดถอยเชิงเส้น (Epoch-Based Training)")
        plt.xlabel("X")
        plt.ylabel("y")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # สร้างโฟลเดอร์ plots ถ้ายังไม่มี
        if not os.path.exists('./plots'):
            os.makedirs('./plots')
            
        plt.savefig('./plots/epoch_based_regression.png')
        plt.close()
        print("บันทึกภาพเส้นถดถอยไปที่ './plots/epoch_based_regression.png'") 