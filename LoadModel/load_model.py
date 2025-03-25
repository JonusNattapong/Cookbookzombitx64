#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import torch
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification
from dotenv import load_dotenv
import argparse

def load_model_from_huggingface(model_name, model_type="model", save_dir=None):
    """
    โหลดโมเดลจาก Hugging Face และบันทึกไว้ที่ Local
    
    Args:
        model_name (str): ชื่อโมเดลจาก Hugging Face (เช่น 'bert-base-uncased')
        model_type (str): ประเภทของโมเดล ('model', 'tokenizer', หรือ 'both')
        save_dir (str): ไดเร็กทอรีที่ต้องการบันทึกโมเดล (ถ้าไม่ระบุจะใช้ชื่อโมเดล)
    
    Returns:
        tuple: (model, tokenizer) หรือ model หรือ tokenizer ตามประเภทที่เลือก
    """
    # สร้างไดเร็กทอรีสำหรับบันทึกโมเดล
    if save_dir is None:
        save_dir = os.path.join(os.getcwd(), model_name.split('/')[-1])
    
    os.makedirs(save_dir, exist_ok=True)
    print(f"จะบันทึกโมเดลไว้ที่: {save_dir}")
    
    model = None
    tokenizer = None
    
    try:
        # โหลดโมเดลตามประเภทที่เลือก
        if model_type in ["model", "both"]:
            print(f"กำลังโหลดโมเดล {model_name}...")
            try:
                # ลองโหลดเป็น sequence classification model ก่อน
                model = AutoModelForSequenceClassification.from_pretrained(model_name)
                print("โหลดเป็น AutoModelForSequenceClassification สำเร็จ")
            except:
                # ถ้าไม่สำเร็จให้โหลดเป็นโมเดลทั่วไป
                model = AutoModel.from_pretrained(model_name)
                print("โหลดเป็น AutoModel สำเร็จ")
            
            # บันทึกโมเดล
            model_path = os.path.join(save_dir, "model")
            model.save_pretrained(model_path)
            print(f"บันทึกโมเดลไว้ที่: {model_path}")
        
        if model_type in ["tokenizer", "both"]:
            print(f"กำลังโหลด tokenizer สำหรับ {model_name}...")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # บันทึก tokenizer
            tokenizer_path = os.path.join(save_dir, "tokenizer")
            tokenizer.save_pretrained(tokenizer_path)
            print(f"บันทึก tokenizer ไว้ที่: {tokenizer_path}")
        
        print("โหลดและบันทึกสำเร็จ!")
        
        # คืนค่าตามประเภทที่เลือก
        if model_type == "both":
            return model, tokenizer
        elif model_type == "model":
            return model
        else:
            return tokenizer
            
    except Exception as e:
        print(f"เกิดข้อผิดพลาด: {str(e)}")
        return None

def load_model_from_local(model_dir, model_type="both"):
    """
    โหลดโมเดลที่บันทึกไว้ในเครื่อง Local
    
    Args:
        model_dir (str): ไดเร็กทอรีที่บันทึกโมเดลไว้
        model_type (str): ประเภทของโมเดล ('model', 'tokenizer', หรือ 'both')
    
    Returns:
        tuple: (model, tokenizer) หรือ model หรือ tokenizer ตามประเภทที่เลือก
    """
    model = None
    tokenizer = None
    
    try:
        if model_type in ["model", "both"]:
            model_path = os.path.join(model_dir, "model")
            if os.path.exists(model_path):
                try:
                    model = AutoModelForSequenceClassification.from_pretrained(model_path)
                    print("โหลดโมเดลเป็น AutoModelForSequenceClassification สำเร็จ")
                except:
                    model = AutoModel.from_pretrained(model_path)
                    print("โหลดโมเดลเป็น AutoModel สำเร็จ")
            else:
                print(f"ไม่พบโมเดลที่ {model_path}")
        
        if model_type in ["tokenizer", "both"]:
            tokenizer_path = os.path.join(model_dir, "tokenizer")
            if os.path.exists(tokenizer_path):
                tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
                print("โหลด tokenizer สำเร็จ")
            else:
                print(f"ไม่พบ tokenizer ที่ {tokenizer_path}")
        
        # คืนค่าตามประเภทที่เลือก
        if model_type == "both":
            return model, tokenizer
        elif model_type == "model":
            return model
        else:
            return tokenizer
            
    except Exception as e:
        print(f"เกิดข้อผิดพลาด: {str(e)}")
        return None

if __name__ == "__main__":
    # โหลด environment variables
    load_dotenv()
    
    # สร้าง argument parser
    parser = argparse.ArgumentParser(description='โหลดโมเดลจาก Hugging Face')
    parser.add_argument('--model', type=str, required=True, help='ชื่อโมเดลจาก Hugging Face (เช่น bert-base-uncased)')
    parser.add_argument('--type', type=str, default='both', choices=['model', 'tokenizer', 'both'], help='ประเภทที่ต้องการโหลด (model, tokenizer, both)')
    parser.add_argument('--save_dir', type=str, default=None, help='ไดเร็กทอรีที่ต้องการบันทึกโมเดล')
    
    args = parser.parse_args()
    
    # โหลดโมเดลจาก Hugging Face
    load_model_from_huggingface(args.model, args.type, args.save_dir) 