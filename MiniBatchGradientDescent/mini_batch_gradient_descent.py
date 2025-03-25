import numpy as np
import matplotlib.pyplot as plt
import os
import time
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import sys

# เพิ่มเส้นทางไปยัง DatasetGenerator เพื่อใช้งาน dataset_generator
sys.path.append('../DatasetGenerator')
try:
    from DatasetGenerator.dataset_generator import load_dataset, generate_linear_dataset
except ImportError:
    print("ไม่สามารถนำเข้าโมดูล dataset_generator ได้")

class LinearRegressionDataset(Dataset):
    """
    คลาสสำหรับจัดการชุดข้อมูลให้อยู่ในรูปแบบที่เหมาะสมสำหรับ PyTorch DataLoader
    """
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class MiniBatchGradientDescent:
    """
    คลาสสำหรับการทำ Mini-Batch Gradient Descent
    
    แนวคิด: ผสมข้อดีของ Vanilla GD และ SGD โดยใช้ batch ขนาดเล็ก (เช่น 32, 64 ตัวอย่าง)
    วิธีการ:
    - แบ่งข้อมูลเป็น batch
    - คำนวณ gradient และอัปเดตน้ำหนักสำหรับแต่ละ batch
    """
    def __init__(self, learning_rate=0.01, epochs=100, batch_size=32, tolerance=1e-6, use_pytorch=True):
        """
        Parameters:
        -----------
        learning_rate : float
            อัตราการเรียนรู้ (learning rate)
        epochs : int
            จำนวนรอบการเรียนรู้ทั้งหมด (epochs)
        batch_size : int
            ขนาด batch (จำนวนตัวอย่างที่ใช้ในการคำนวณ gradient แต่ละครั้ง)
        tolerance : float
            ค่าความคลาดเคลื่อนที่ยอมรับได้สำหรับการหยุดการเรียนรู้
        use_pytorch : bool
            ใช้ PyTorch DataLoader หรือไม่ (True) หรือใช้การจัดการ batch แบบปกติ (False)
        """
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.tolerance = tolerance
        self.use_pytorch = use_pytorch
        self.weights = None
        self.bias = None
        self.loss_history = []
        self.epoch_loss_history = []
        
        # ตรวจสอบว่าสามารถใช้งาน PyTorch ได้หรือไม่
        if use_pytorch:
            try:
                import torch
                self.torch_available = True
            except ImportError:
                print("ไม่พบ PyTorch กำลังใช้การจัดการ batch แบบปกติแทน")
                self.torch_available = False
                self.use_pytorch = False
        else:
            self.torch_available = False
    
    def initialize_parameters(self, n_features):
        """
        กำหนดค่าเริ่มต้นของพารามิเตอร์
        """
        if self.use_pytorch and self.torch_available:
            self.weights = torch.zeros(n_features, requires_grad=True)
            self.bias = torch.zeros(1, requires_grad=True)
        else:
            self.weights = np.zeros(n_features)
            self.bias = 0
    
    def compute_predictions(self, X):
        """
        คำนวณค่าทำนาย
        """
        if self.use_pytorch and self.torch_available and isinstance(X, torch.Tensor):
            return torch.matmul(X, self.weights) + self.bias
        else:
            return np.dot(X, self.weights) + self.bias
    
    def compute_loss(self, y_true, y_pred):
        """
        คำนวณค่าความสูญเสีย (Mean Squared Error)
        """
        if self.use_pytorch and self.torch_available and isinstance(y_true, torch.Tensor) and isinstance(y_pred, torch.Tensor):
            return torch.mean((y_pred - y_true) ** 2) / 2
        else:
            m = len(y_true)
            return np.sum((y_pred - y_true) ** 2) / (2 * m)
    
    def compute_gradients(self, X, y_true, y_pred):
        """
        คำนวณค่า gradient ของ loss function
        """
        if self.use_pytorch and self.torch_available and isinstance(X, torch.Tensor) and isinstance(y_true, torch.Tensor) and isinstance(y_pred, torch.Tensor):
            # PyTorch จะคำนวณ gradient ให้อัตโนมัติ
            loss = self.compute_loss(y_true, y_pred)
            loss.backward()
            return self.weights.grad, self.bias.grad
        else:
            m = len(y_true)
            dw = np.dot(X.T, (y_pred - y_true)) / m
            db = np.sum(y_pred - y_true) / m
            return dw, db
    
    def update_parameters(self, dw, db):
        """
        อัปเดตค่าพารามิเตอร์
        """
        if self.use_pytorch and self.torch_available and isinstance(dw, torch.Tensor) and isinstance(db, torch.Tensor):
            with torch.no_grad():
                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db
                
            # ล้าง gradient
            self.weights.grad.zero_()
            self.bias.grad.zero_()
        else:
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
    
    def create_batches(self, X, y):
        """
        สร้าง batches จากข้อมูลแบบปกติ (ไม่ใช้ PyTorch)
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
        ฝึกโมเดลด้วย Mini-Batch Gradient Descent
        """
        # กำหนดค่าเริ่มต้น
        n_samples, n_features = X.shape
        self.initialize_parameters(n_features)
        
        # เตรียมข้อมูลสำหรับ PyTorch
        if self.use_pytorch and self.torch_available:
            dataset = LinearRegressionDataset(X, y)
            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # วนรอบการเรียนรู้ (epochs)
        for epoch in range(self.epochs):
            epoch_start_time = time.time()
            
            # เก็บค่า loss ของแต่ละ batch ในรอบนี้
            batch_losses = []
            
            if self.use_pytorch and self.torch_available:
                # ใช้ PyTorch DataLoader
                for batch_X, batch_y in dataloader:
                    # คำนวณค่าทำนาย
                    batch_y_pred = self.compute_predictions(batch_X)
                    
                    # คำนวณ gradient
                    dw, db = self.compute_gradients(batch_X, batch_y, batch_y_pred)
                    
                    # อัปเดตพารามิเตอร์
                    self.update_parameters(dw, db)
                    
                    # คำนวณค่าความสูญเสียสำหรับ batch นี้
                    batch_loss = self.compute_loss(batch_y, batch_y_pred).item()
                    batch_losses.append(batch_loss)
                    self.loss_history.append(batch_loss)
            else:
                # ใช้การจัดการ batch แบบปกติ
                batches = self.create_batches(X, y)
                
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
        
        # แปลงพารามิเตอร์กลับเป็น numpy array ถ้าใช้ PyTorch
        if self.use_pytorch and self.torch_available:
            self.weights = self.weights.detach().numpy()
            self.bias = self.bias.detach().numpy().item()
        
        return self
    
    def predict(self, X):
        """
        ทำนายผลลัพธ์
        """
        if self.use_pytorch and self.torch_available and not isinstance(X, torch.Tensor):
            X = torch.FloatTensor(X)
            predictions = self.compute_predictions(X)
            return predictions.detach().numpy()
        else:
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
            
        plt.savefig('./plots/mini_batch_loss_history.png')
        plt.close()
        
        print("บันทึกกราฟประวัติค่าความสูญเสียไปที่ './plots/mini_batch_loss_history.png'")
    
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
        plt.title("เส้นถดถอยเชิงเส้น (Mini-Batch GD)")
        plt.xlabel("X")
        plt.ylabel("y")
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # สร้างโฟลเดอร์เพื่อบันทึกรูปภาพ
        if not os.path.exists('./plots'):
            os.makedirs('./plots')
            
        plt.savefig('./plots/mini_batch_regression_line.png')
        plt.close()
        
        print("บันทึกกราฟเส้นถดถอยไปที่ './plots/mini_batch_regression_line.png'")

if __name__ == "__main__":
    # โหลดหรือสร้างชุดข้อมูล
    try:
        X, y = load_dataset()
        print("โหลดชุดข้อมูลสำเร็จ")
    except:
        print("ไม่พบชุดข้อมูล กำลังสร้างชุดข้อมูลใหม่...")
        X, y = generate_linear_dataset(n_samples=100, n_features=1, noise=15.0)
    
    # สร้างและฝึกโมเดล (ใช้ PyTorch DataLoader)
    try:
        model_pytorch = MiniBatchGradientDescent(learning_rate=0.1, epochs=50, batch_size=32, use_pytorch=True)
        model_pytorch.fit(X, y)
        
        # แสดงผลลัพธ์
        y_pred = model_pytorch.predict(X)
        mse = model_pytorch.compute_loss(y, y_pred)
        print("\nผลลัพธ์โมเดลที่ใช้ PyTorch DataLoader:")
        print(f"ค่าน้ำหนัก (Weights): {model_pytorch.weights}")
        print(f"ค่า Bias: {model_pytorch.bias}")
        print(f"ค่าความสูญเสีย (MSE): {mse:.6f}")
        
        # แสดงกราฟ
        model_pytorch.plot_loss_history()
        model_pytorch.plot_regression_line(X, y)
    except Exception as e:
        print(f"\nเกิดข้อผิดพลาดในการใช้งาน PyTorch: {e}")
    
    # สร้างและฝึกโมเดล (ไม่ใช้ PyTorch DataLoader)
    model_numpy = MiniBatchGradientDescent(learning_rate=0.1, epochs=50, batch_size=32, use_pytorch=False)
    model_numpy.fit(X, y)
    
    # แสดงผลลัพธ์
    y_pred = model_numpy.predict(X)
    mse = model_numpy.compute_loss(y, y_pred)
    print("\nผลลัพธ์โมเดลที่ไม่ใช้ PyTorch DataLoader:")
    print(f"ค่าน้ำหนัก (Weights): {model_numpy.weights}")
    print(f"ค่า Bias: {model_numpy.bias}")
    print(f"ค่าความสูญเสีย (MSE): {mse:.6f}")
    
    # แสดงกราฟ
    model_numpy.plot_loss_history()
    model_numpy.plot_regression_line(X, y) 