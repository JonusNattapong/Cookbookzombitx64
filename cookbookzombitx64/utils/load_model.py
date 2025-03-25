"""
โมดูลสำหรับโหลดโมเดลเอไอจากไฟล์หรือจากแหล่งข้อมูลออนไลน์
"""

import os
import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

def load_torch_model(model_path, device=None):
    """
    โหลดโมเดล PyTorch
    
    Args:
        model_path (str): พาธของโมเดลที่จะโหลด
        device (str, optional): อุปกรณ์ที่จะโหลดโมเดล (cuda หรือ cpu)
    
    Returns:
        model: โมเดลที่โหลดแล้ว
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
    model = torch.load(model_path, map_location=device)
    model.to(device)
    return model

def load_huggingface_model(model_name, model_type="base", device=None):
    """
    โหลดโมเดลจาก Hugging Face
    
    Args:
        model_name (str): ชื่อโมเดลใน Hugging Face
        model_type (str): ประเภทของโมเดล ('base' หรือ 'causal_lm')
        device (str, optional): อุปกรณ์ที่จะโหลดโมเดล (cuda หรือ cpu)
    
    Returns:
        tuple: (model, tokenizer) โมเดลและ tokenizer ที่โหลดแล้ว
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
    if model_type == "causal_lm":
        model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    else:
        model = AutoModel.from_pretrained(model_name).to(device)
        
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

def load_checkpoint(model, checkpoint_path, device=None):
    """
    โหลด checkpoint ลงในโมเดลที่มีอยู่แล้ว
    
    Args:
        model: โมเดลที่จะโหลด checkpoint
        checkpoint_path (str): พาธของ checkpoint
        device (str, optional): อุปกรณ์ที่จะโหลดโมเดล (cuda หรือ cpu)
    
    Returns:
        model: โมเดลที่โหลด checkpoint แล้ว
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model 