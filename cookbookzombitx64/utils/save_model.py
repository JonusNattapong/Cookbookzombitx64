"""
โมดูลสำหรับบันทึกโมเดลเอไอลงในไฟล์
"""

import os
import torch
import json
import time
from pathlib import Path

def save_torch_model(model, path, optimizer=None, epoch=None, loss=None, metadata=None):
    """
    บันทึกโมเดล PyTorch พร้อมข้อมูลประกอบ
    
    Args:
        model: โมเดลที่จะบันทึก
        path (str): พาธที่จะบันทึกโมเดล
        optimizer (optional): optimizer ที่ใช้เทรนโมเดล
        epoch (int, optional): epoch ปัจจุบัน
        loss (float, optional): ค่า loss ล่าสุด
        metadata (dict, optional): ข้อมูลประกอบอื่นๆ
        
    Returns:
        str: พาธของไฟล์ที่บันทึก
    """
    # สร้างไดเรกทอรีถ้ายังไม่มี
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # เตรียมข้อมูลที่จะบันทึก
    save_dict = {
        'model_state_dict': model.state_dict(),
        'timestamp': time.time()
    }
    
    if optimizer is not None:
        save_dict['optimizer_state_dict'] = optimizer.state_dict()
    
    if epoch is not None:
        save_dict['epoch'] = epoch
        
    if loss is not None:
        save_dict['loss'] = loss
        
    if metadata is not None:
        save_dict['metadata'] = metadata
    
    # บันทึกโมเดล
    torch.save(save_dict, path)
    
    return path

def save_with_config(model, base_path, config, tokenizer=None):
    """
    บันทึกโมเดลพร้อมไฟล์การกำหนดค่า
    
    Args:
        model: โมเดลที่จะบันทึก
        base_path (str): พาธพื้นฐานที่จะบันทึกไฟล์
        config (dict): การกำหนดค่าของโมเดล
        tokenizer (optional): tokenizer ที่ใช้กับโมเดล
        
    Returns:
        str: พาธของไดเรกทอรีที่บันทึก
    """
    # สร้างไดเรกทอรี
    path = Path(base_path)
    path.mkdir(parents=True, exist_ok=True)
    
    # บันทึกโมเดล
    model_path = path / "model.pt"
    torch.save(model.state_dict(), model_path)
    
    # บันทึกการกำหนดค่า
    config_path = path / "config.json"
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=4)
    
    # บันทึก tokenizer ถ้ามี
    if tokenizer is not None:
        tokenizer.save_pretrained(path)
    
    return str(path) 