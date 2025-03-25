import numpy as np
import requests
import json
import os

class MistralAIDatasetGenerator:
    def __init__(self, api_key=None, dataset_name="dataset"):
        """
        เริ่มต้นตัวสร้างชุดข้อมูลด้วย Mistral AI API
        
        Parameters:
        -----------
        api_key : str, optional
            API key สำหรับ Mistral AI ถ้าไม่ระบุจะใช้จาก environment variable
        dataset_name : str, optional
            ชื่อชุดข้อมูล (default: "dataset")
        """
        self.api_key = api_key or os.getenv('MISTRAL_API_KEY')
        if not self.api_key:
            raise ValueError("กรุณาระบุ API key สำหรับ Mistral AI")
        
        self.dataset_name = dataset_name
        self.api_url = "https://api.mistral.ai/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def generate_linear_dataset(self, n_samples=100, n_features=1, noise=15.0):
        """
        สร้างชุดข้อมูลเชิงเส้นโดยใช้ Mistral AI API
        
        Parameters:
        -----------
        n_samples : int
            จำนวนตัวอย่างข้อมูล
        n_features : int
            จำนวนคุณลักษณะ
        noise : float
            ระดับความรบกวนในข้อมูล
            
        Returns:
        --------
        X : numpy array
            ข้อมูลคุณลักษณะ
        y : numpy array
            ข้อมูลเป้าหมาย
        """
        # สร้าง prompt สำหรับ Mistral AI
        prompt = f"""
        สร้างชุดข้อมูลเชิงเส้นที่มีคุณสมบัติดังนี้:
        - จำนวนตัวอย่าง: {n_samples}
        - จำนวนคุณลักษณะ: {n_features}
        - ระดับความรบกวน: {noise}
        
        กรุณาสร้างข้อมูลในรูปแบบ JSON ที่มี:
        1. X: array ของ features
        2. y: array ของ target values
        3. true_weights: array ของน้ำหนักจริง
        4. true_bias: ค่า bias จริง
        """
        
        # ส่งคำขอไปยัง Mistral AI API
        payload = {
            "model": "mistral-tiny",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7
        }
        
        try:
            response = requests.post(self.api_url, headers=self.headers, json=payload)
            response.raise_for_status()
            result = response.json()
            
            # แปลงผลลัพธ์เป็น numpy arrays
            data = json.loads(result['choices'][0]['message']['content'])
            X = np.array(data['X'])
            y = np.array(data['y'])
            
            # บันทึกข้อมูล
            self.save_dataset(X, y, data['true_weights'], data['true_bias'])
            
            return X, y
            
        except Exception as e:
            print(f"เกิดข้อผิดพลาดในการสร้างชุดข้อมูล: {e}")
            # ถ้าเกิดข้อผิดพลาด ให้สร้างข้อมูลแบบสุ่มแทน
            return self._generate_random_dataset(n_samples, n_features, noise)
    
    def _generate_random_dataset(self, n_samples, n_features, noise):
        """
        สร้างชุดข้อมูลเชิงเส้นแบบสุ่ม (fallback method)
        """
        # สร้างน้ำหนักและ bias แบบสุ่ม
        true_weights = np.random.randn(n_features)
        true_bias = np.random.randn()
        
        # สร้างข้อมูล X
        X = np.random.randn(n_samples, n_features)
        
        # สร้างข้อมูล y
        y = np.dot(X, true_weights) + true_bias
        
        # เพิ่มความรบกวน
        y += np.random.normal(0, noise, n_samples)
        
        # บันทึกข้อมูล
        self.save_dataset(X, y, true_weights, true_bias)
        
        return X, y
    
    def save_dataset(self, X, y, true_weights, true_bias):
        """
        บันทึกชุดข้อมูลลงไฟล์
        """
        data = {
            'X': X,
            'y': y,
            'true_weights': true_weights,
            'true_bias': true_bias
        }
        
        # สร้างโฟลเดอร์ data ถ้ายังไม่มี
        if not os.path.exists('./data'):
            os.makedirs('./data')
            
        # บันทึกข้อมูล
        filename = f'./data/{self.dataset_name}.npz'
        np.savez(filename, **data)
        print(f"บันทึกชุดข้อมูลไปที่ '{filename}'")
    
    def load_dataset(self):
        """
        โหลดชุดข้อมูลจากไฟล์
        """
        try:
            filename = f'./data/{self.dataset_name}.npz'
            data = np.load(filename)
            return data['X'], data['y']
        except FileNotFoundError:
            print(f"ไม่พบไฟล์ชุดข้อมูล '{filename}' กำลังสร้างชุดข้อมูลใหม่...")
            return self.generate_linear_dataset()
    
    def visualize_dataset(self, X, y, save_path=None):
        """
        แสดงภาพชุดข้อมูล
        """
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 6))
        plt.scatter(X, y, alpha=0.7)
        plt.title(f"ชุดข้อมูลสำหรับการถดถอยเชิงเส้น ({self.dataset_name})")
        plt.xlabel("X")
        plt.ylabel("y")
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

# ฟังก์ชันสำหรับใช้งานง่าย
def generate_linear_dataset(n_samples=100, n_features=1, noise=15.0, dataset_name="dataset"):
    """
    สร้างชุดข้อมูลเชิงเส้น
    """
    generator = MistralAIDatasetGenerator(dataset_name=dataset_name)
    return generator.generate_linear_dataset(n_samples, n_features, noise)

def load_dataset(dataset_name="dataset"):
    """
    โหลดชุดข้อมูล
    """
    generator = MistralAIDatasetGenerator(dataset_name=dataset_name)
    return generator.load_dataset()

def visualize_dataset(X, y, save_path=None, dataset_name="dataset"):
    """
    แสดงภาพชุดข้อมูล
    """
    generator = MistralAIDatasetGenerator(dataset_name=dataset_name)
    generator.visualize_dataset(X, y, save_path)

def save_dataset(X, y, true_weights, true_bias, dataset_name="dataset"):
    """
    บันทึกชุดข้อมูล
    """
    generator = MistralAIDatasetGenerator(dataset_name=dataset_name)
    generator.save_dataset(X, y, true_weights, true_bias) 