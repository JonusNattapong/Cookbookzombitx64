"""
โมดูลสำหรับการเทรนโมเดลแบบ Epoch-Based (การเทรนโดยแบ่งตามรอบหรือยุค)
"""

import os
import time
import torch
import numpy as np
from typing import Callable, Dict, List, Optional, Union, Tuple
from torch.utils.data import DataLoader, Dataset
from torch.optim import Optimizer
from torch.nn import Module

class EpochTrainer:
    """คลาสสำหรับการเทรนโมเดลแบบ Epoch-Based"""
    
    def __init__(
        self, 
        model: Module,
        optimizer: Optimizer,
        loss_fn: Callable,
        device: Optional[str] = None,
        scheduler: Optional[object] = None
    ):
        """
        กำหนดค่าเริ่มต้นสำหรับการเทรนแบบ Epoch-Based
        
        Args:
            model (Module): โมเดลที่จะทำการเทรน
            optimizer (Optimizer): ตัวปรับค่าพารามิเตอร์ของโมเดล
            loss_fn (Callable): ฟังก์ชันสำหรับคำนวณค่าความสูญเสีย (loss)
            device (str, optional): อุปกรณ์ที่ใช้ในการคำนวณ ('cuda', 'cpu') ถ้าไม่ระบุจะเลือกอัตโนมัติ
            scheduler (object, optional): ตัวปรับค่า learning rate
        """
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        
        # กำหนดอุปกรณ์ที่ใช้ในการคำนวณ
        if device:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
        # ย้ายโมเดลไปยังอุปกรณ์
        self.model.to(self.device)
        
        # ประวัติการเทรน
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'metrics': {}
        }
        
    def train_epoch(self, train_loader: DataLoader, metrics: Optional[Dict[str, Callable]] = None) -> Dict[str, float]:
        """
        เทรนโมเดลเป็นเวลาหนึ่ง epoch
        
        Args:
            train_loader (DataLoader): DataLoader สำหรับชุดข้อมูลฝึก
            metrics (Dict[str, Callable], optional): Dictionary ของเมทริกสำหรับวัดประสิทธิภาพ
            
        Returns:
            Dict[str, float]: Dictionary ของค่าเฉลี่ย loss และเมทริกอื่นๆ
        """
        self.model.train()  # ตั้งค่าโมเดลเป็นโหมดเทรน
        
        epoch_loss = 0.0
        epoch_metrics = {name: 0.0 for name in metrics.keys()} if metrics else {}
        num_batches = len(train_loader)
        
        # วนลูปผ่านแต่ละแบตช์
        for inputs, targets in train_loader:
            # ย้ายข้อมูลไปยังอุปกรณ์
            inputs = self._move_to_device(inputs)
            targets = self._move_to_device(targets)
            
            # เคลียร์เกรเดียนต์ที่คำนวณไว้ก่อนหน้า
            self.optimizer.zero_grad()
            
            # คำนวณการทำนาย
            outputs = self.model(inputs)
            
            # คำนวณค่า loss
            loss = self.loss_fn(outputs, targets)
            
            # คำนวณเกรเดียนต์และปรับค่าพารามิเตอร์
            loss.backward()
            self.optimizer.step()
            
            # ปรับค่า learning rate (ถ้ามี scheduler)
            if self.scheduler and hasattr(self.scheduler, 'step'):
                self.scheduler.step()
            
            # เก็บค่า loss
            epoch_loss += loss.item()
            
            # คำนวณเมทริกเพิ่มเติม (ถ้ามี)
            if metrics:
                for name, metric_fn in metrics.items():
                    epoch_metrics[name] += metric_fn(outputs, targets).item()
        
        # คำนวณค่าเฉลี่ย
        epoch_loss /= num_batches
        for name in epoch_metrics.keys():
            epoch_metrics[name] /= num_batches
        
        # รวมผลลัพธ์
        results = {'loss': epoch_loss}
        results.update(epoch_metrics)
        
        return results
    
    def validate(self, val_loader: DataLoader, metrics: Optional[Dict[str, Callable]] = None) -> Dict[str, float]:
        """
        ประเมินผลโมเดลกับชุดข้อมูลตรวจสอบ
        
        Args:
            val_loader (DataLoader): DataLoader สำหรับชุดข้อมูลตรวจสอบ
            metrics (Dict[str, Callable], optional): Dictionary ของเมทริกสำหรับวัดประสิทธิภาพ
            
        Returns:
            Dict[str, float]: Dictionary ของค่าเฉลี่ย loss และเมทริกอื่นๆ
        """
        self.model.eval()  # ตั้งค่าโมเดลเป็นโหมดประเมินผล
        
        epoch_loss = 0.0
        epoch_metrics = {name: 0.0 for name in metrics.keys()} if metrics else {}
        num_batches = len(val_loader)
        
        # ไม่คำนวณเกรเดียนต์
        with torch.no_grad():
            # วนลูปผ่านแต่ละแบตช์
            for inputs, targets in val_loader:
                # ย้ายข้อมูลไปยังอุปกรณ์
                inputs = self._move_to_device(inputs)
                targets = self._move_to_device(targets)
                
                # คำนวณการทำนาย
                outputs = self.model(inputs)
                
                # คำนวณค่า loss
                loss = self.loss_fn(outputs, targets)
                
                # เก็บค่า loss
                epoch_loss += loss.item()
                
                # คำนวณเมทริกเพิ่มเติม (ถ้ามี)
                if metrics:
                    for name, metric_fn in metrics.items():
                        epoch_metrics[name] += metric_fn(outputs, targets).item()
        
        # คำนวณค่าเฉลี่ย
        epoch_loss /= num_batches
        for name in epoch_metrics.keys():
            epoch_metrics[name] /= num_batches
        
        # รวมผลลัพธ์
        results = {'loss': epoch_loss}
        results.update(epoch_metrics)
        
        return results
    
    def fit(
        self, 
        train_loader: DataLoader, 
        num_epochs: int, 
        val_loader: Optional[DataLoader] = None,
        metrics: Optional[Dict[str, Callable]] = None,
        checkpoint_dir: Optional[str] = None,
        early_stopping: Optional[int] = None,
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        """
        เทรนโมเดลเป็นเวลาหลาย epoch
        
        Args:
            train_loader (DataLoader): DataLoader สำหรับชุดข้อมูลฝึก
            num_epochs (int): จำนวน epoch ที่จะเทรน
            val_loader (DataLoader, optional): DataLoader สำหรับชุดข้อมูลตรวจสอบ
            metrics (Dict[str, Callable], optional): Dictionary ของเมทริกสำหรับวัดประสิทธิภาพ
            checkpoint_dir (str, optional): ไดเรกทอรีสำหรับบันทึก checkpoint
            early_stopping (int, optional): จำนวน epoch ที่รอก่อนหยุดการเทรนถ้า val_loss ไม่ลดลง
            verbose (bool): แสดงผลลัพธ์ระหว่างการเทรนหรือไม่
            
        Returns:
            Dict[str, List[float]]: ประวัติการเทรน
        """
        if checkpoint_dir and not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
            
        best_val_loss = float('inf')
        early_stopping_counter = 0
        
        # เริ่มการเทรน
        for epoch in range(num_epochs):
            start_time = time.time()
            
            # เทรนหนึ่ง epoch
            train_results = self.train_epoch(train_loader, metrics)
            
            # ประเมินผลกับชุดข้อมูลตรวจสอบ (ถ้ามี)
            val_results = {}
            if val_loader:
                val_results = self.validate(val_loader, metrics)
            
            # ปรับค่า learning rate สำหรับ schedulers ที่ปรับตาม epoch
            if self.scheduler and not hasattr(self.scheduler, 'step'):
                if hasattr(self.scheduler, 'step_on_validation'):
                    self.scheduler.step(val_results['loss'])
                else:
                    self.scheduler.step()
            
            # บันทึกประวัติ
            self.history['train_loss'].append(train_results['loss'])
            if val_loader:
                self.history['val_loss'].append(val_results['loss'])
                
            # บันทึกเมทริกอื่นๆ
            if metrics:
                for name in metrics.keys():
                    if name not in self.history['metrics']:
                        self.history['metrics'][name] = {'train': [], 'val': []}
                    
                    self.history['metrics'][name]['train'].append(train_results[name])
                    if val_loader:
                        self.history['metrics'][name]['val'].append(val_results[name])
            
            # แสดงผลลัพธ์
            if verbose:
                epoch_time = time.time() - start_time
                self._print_epoch_results(epoch + 1, num_epochs, train_results, val_results, epoch_time)
            
            # บันทึก checkpoint สำหรับโมเดลที่ดีที่สุด
            if val_loader and val_results['loss'] < best_val_loss:
                best_val_loss = val_results['loss']
                early_stopping_counter = 0
                
                if checkpoint_dir:
                    self.save_checkpoint(os.path.join(checkpoint_dir, 'best_model.pth'))
                    
            # Early stopping
            elif val_loader and early_stopping:
                early_stopping_counter += 1
                if early_stopping_counter >= early_stopping:
                    if verbose:
                        print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                    break
        
        # บันทึก checkpoint สุดท้าย
        if checkpoint_dir:
            self.save_checkpoint(os.path.join(checkpoint_dir, 'last_model.pth'))
            
        return self.history
    
    def save_checkpoint(self, path: str):
        """
        บันทึก checkpoint ของโมเดล
        
        Args:
            path (str): พาธที่จะบันทึก checkpoint
        """
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
            
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: str):
        """
        โหลด checkpoint ของโมเดล
        
        Args:
            path (str): พาธที่มี checkpoint
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint['history']
        
        if 'scheduler_state_dict' in checkpoint and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    def _move_to_device(self, tensor_or_dict):
        """
        ย้ายข้อมูลไปยังอุปกรณ์ที่กำหนด
        
        Args:
            tensor_or_dict: Tensor หรือ Dictionary ของ Tensor
            
        Returns:
            tensor_or_dict ที่ย้ายไปยังอุปกรณ์แล้ว
        """
        if isinstance(tensor_or_dict, torch.Tensor):
            return tensor_or_dict.to(self.device)
        elif isinstance(tensor_or_dict, dict):
            return {k: self._move_to_device(v) for k, v in tensor_or_dict.items()}
        else:
            return tensor_or_dict
    
    def _print_epoch_results(self, epoch: int, total_epochs: int, train_results: Dict[str, float], 
                            val_results: Dict[str, float], time_taken: float):
        """
        แสดงผลลัพธ์ของแต่ละ epoch
        
        Args:
            epoch (int): epoch ปัจจุบัน
            total_epochs (int): จำนวน epoch ทั้งหมด
            train_results (Dict[str, float]): ผลลัพธ์จากการเทรน
            val_results (Dict[str, float]): ผลลัพธ์จากการตรวจสอบ
            time_taken (float): เวลาที่ใช้ (วินาที)
        """
        output = f"Epoch {epoch}/{total_epochs} - {time_taken:.2f}s - loss: {train_results['loss']:.4f}"
        
        # เพิ่มเมทริกอื่นๆ สำหรับการเทรน
        for name, value in train_results.items():
            if name != 'loss':
                output += f" - {name}: {value:.4f}"
        
        # เพิ่มผลลัพธ์จากการตรวจสอบ (ถ้ามี)
        if val_results:
            output += f" - val_loss: {val_results['loss']:.4f}"
            
            for name, value in val_results.items():
                if name != 'loss':
                    output += f" - val_{name}: {value:.4f}"
        
        print(output)
        
class MetricTracker:
    """คลาสสำหรับติดตามเมทริกการเทรน"""
    
    @staticmethod
    def accuracy(outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """คำนวณความแม่นยำ (accuracy)"""
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == targets).sum().item()
        return torch.tensor(correct / targets.size(0))
    
    @staticmethod
    def precision(outputs: torch.Tensor, targets: torch.Tensor, average: bool = True) -> torch.Tensor:
        """คำนวณความแม่นยำ (precision)"""
        _, predicted = torch.max(outputs.data, 1)
        
        # คำนวณ precision สำหรับแต่ละคลาส
        classes = torch.unique(targets)
        precision_values = []
        
        for c in classes:
            # True positives: ที่ทำนายถูกต้องว่าเป็นคลาส c
            true_positives = ((predicted == c) & (targets == c)).sum().float()
            
            # ทั้งหมดที่ทำนายว่าเป็นคลาส c
            predicted_positives = (predicted == c).sum().float()
            
            # ป้องกัน division by zero
            if predicted_positives > 0:
                precision = true_positives / predicted_positives
            else:
                precision = torch.tensor(0.0, device=outputs.device)
                
            precision_values.append(precision)
        
        # คืนค่าเฉลี่ยหรือรายคลาส
        if average:
            return torch.mean(torch.stack(precision_values))
        else:
            return torch.stack(precision_values)
    
    @staticmethod
    def recall(outputs: torch.Tensor, targets: torch.Tensor, average: bool = True) -> torch.Tensor:
        """คำนวณความครบถ้วน (recall)"""
        _, predicted = torch.max(outputs.data, 1)
        
        # คำนวณ recall สำหรับแต่ละคลาส
        classes = torch.unique(targets)
        recall_values = []
        
        for c in classes:
            # True positives: ที่ทำนายถูกต้องว่าเป็นคลาส c
            true_positives = ((predicted == c) & (targets == c)).sum().float()
            
            # ทั้งหมดที่เป็นคลาส c จริงๆ
            actual_positives = (targets == c).sum().float()
            
            # ป้องกัน division by zero
            if actual_positives > 0:
                recall = true_positives / actual_positives
            else:
                recall = torch.tensor(0.0, device=outputs.device)
                
            recall_values.append(recall)
        
        # คืนค่าเฉลี่ยหรือรายคลาส
        if average:
            return torch.mean(torch.stack(recall_values))
        else:
            return torch.stack(recall_values)
    
    @staticmethod
    def f1_score(outputs: torch.Tensor, targets: torch.Tensor, average: bool = True) -> torch.Tensor:
        """คำนวณค่า F1 (F1 score)"""
        precision = MetricTracker.precision(outputs, targets, average=False)
        recall = MetricTracker.recall(outputs, targets, average=False)
        
        # ป้องกัน division by zero
        denominator = precision + recall
        f1 = torch.where(denominator > 0, 2 * precision * recall / denominator, torch.zeros_like(denominator))
        
        # คืนค่าเฉลี่ยหรือรายคลาส
        if average:
            return torch.mean(f1)
        else:
            return f1
    
    @staticmethod
    def mean_squared_error(outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """คำนวณค่าเฉลี่ยความคลาดเคลื่อนกำลังสอง (MSE)"""
        return torch.mean((outputs - targets) ** 2)
    
    @staticmethod
    def root_mean_squared_error(outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """คำนวณค่ารากที่สองของค่าเฉลี่ยความคลาดเคลื่อนกำลังสอง (RMSE)"""
        return torch.sqrt(MetricTracker.mean_squared_error(outputs, targets))
    
    @staticmethod
    def mean_absolute_error(outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """คำนวณค่าเฉลี่ยความคลาดเคลื่อนสัมบูรณ์ (MAE)"""
        return torch.mean(torch.abs(outputs - targets)) 