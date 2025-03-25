#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ตัวอย่างการใช้งาน load_model.py สำหรับโหลดโมเดลจาก Hugging Face มาไว้ที่ Local
"""

from load_model import load_model_from_huggingface, load_model_from_local

def example_load_from_huggingface():
    """
    ตัวอย่างการโหลดโมเดลจาก Hugging Face มาไว้ที่ Local
    """
    # ตัวอย่างที่ 1: โหลดทั้งโมเดลและ tokenizer
    model_name = "bert-base-uncased"
    model, tokenizer = load_model_from_huggingface(model_name, model_type="both")
    
    # ตัวอย่างที่ 2: โหลดเฉพาะโมเดล
    # model = load_model_from_huggingface("gpt2", model_type="model")
    
    # ตัวอย่างที่ 3: โหลดเฉพาะ tokenizer
    # tokenizer = load_model_from_huggingface("roberta-base", model_type="tokenizer")
    
    # ตัวอย่างที่ 4: ระบุไดเร็กทอรีที่จะบันทึก
    # model = load_model_from_huggingface("distilbert-base-uncased", save_dir="./my_models/distilbert")
    
    return model, tokenizer

def example_load_from_local():
    """
    ตัวอย่างการโหลดโมเดลที่บันทึกไว้ใน Local
    """
    # ตัวอย่างที่ 1: โหลดทั้งโมเดลและ tokenizer
    model_dir = "./bert-base-uncased"  # ไดเร็กทอรีที่บันทึกโมเดลไว้
    model, tokenizer = load_model_from_local(model_dir)
    
    # ตัวอย่างที่ 2: โหลดเฉพาะโมเดล
    # model = load_model_from_local("./gpt2", model_type="model")
    
    # ตัวอย่างที่ 3: โหลดเฉพาะ tokenizer
    # tokenizer = load_model_from_local("./roberta-base", model_type="tokenizer")
    
    return model, tokenizer

def example_usage():
    """
    ตัวอย่างการใช้งานโมเดลและ tokenizer
    """
    # โหลดโมเดลจาก Hugging Face หรือจาก Local
    model, tokenizer = example_load_from_local()  # หรือใช้ example_load_from_huggingface()
    
    # ตัวอย่างการใช้งาน tokenizer และโมเดลสำหรับทำนาย
    text = "ฉันชอบเรียนรู้เกี่ยวกับ AI และ machine learning"
    
    # แปลงข้อความเป็น input ของโมเดล
    inputs = tokenizer(text, return_tensors="pt")
    
    # ประมวลผลด้วยโมเดล
    with torch.no_grad():
        outputs = model(**inputs)
    
    print(f"Input text: {text}")
    print(f"Model output shape: {outputs.last_hidden_state.shape}")
    
    return outputs

if __name__ == "__main__":
    # ตัวอย่างการโหลดและใช้งานโมเดล
    try:
        import torch
        example_usage()
    except Exception as e:
        print(f"เกิดข้อผิดพลาด: {str(e)}") 