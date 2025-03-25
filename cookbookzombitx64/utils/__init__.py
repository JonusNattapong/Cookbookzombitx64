"""
โมดูลที่รวบรวมฟังก์ชันช่วยเหลือต่างๆ สำหรับการทำงานกับโมเดลเอไอ
"""

from . import load_model
from . import save_model
from . import hyperparameter_tuning

__all__ = ["load_model", "save_model", "hyperparameter_tuning"]

def get_device():
    """
    ตรวจสอบและคืนค่าอุปกรณ์ที่ใช้ในการคำนวณ
    
    Returns:
        str: ชื่ออุปกรณ์ ('cuda' หรือ 'cpu')
    """
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu" 