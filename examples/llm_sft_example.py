#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ตัวอย่างการใช้งานโมดูล LLM SFT (Supervised Fine-Tuning)
"""

import os
import torch
from datasets import load_dataset
from dotenv import load_dotenv
from cookbookzombitx64.llm.sft import LLMSFTTrainer

# โหลดค่าตัวแปรสภาพแวดล้อม
load_dotenv()

def main():
    """ฟังก์ชันหลักสำหรับตัวอย่างการใช้งาน"""
    
    # ตั้งค่าโมเดลที่จะใช้
    model_name = "gpt2"  # ใช้โมเดล GPT-2 ขนาดเล็กเพื่อเป็นตัวอย่าง
    
    print(f"เริ่มต้นการใช้งาน LLM SFT Trainer ด้วยโมเดล {model_name}")
    
    # สร้างออบเจ็กต์ LLMSFTTrainer
    trainer = LLMSFTTrainer(model_name)
    
    print(f"โมเดลทำงานบนอุปกรณ์: {trainer.device}")
    
    # โหลดชุดข้อมูลตัวอย่าง (ใช้ชุดข้อมูล tiny_shakespeare)
    dataset = load_dataset("tiny_shakespeare", split="train")
    print(f"โหลดชุดข้อมูล: {dataset}")
    
    # เตรียมชุดข้อมูล
    prepared_dataset = trainer.prepare_dataset(dataset, text_column="text", max_length=128)
    print(f"เตรียมชุดข้อมูลเสร็จสิ้น: {prepared_dataset}")
    
    # แบ่งชุดข้อมูลสำหรับการฝึกและการประเมินผล
    train_size = int(0.9 * len(prepared_dataset))
    train_dataset = prepared_dataset.select(range(train_size))
    eval_dataset = prepared_dataset.select(range(train_size, len(prepared_dataset)))
    
    # เทรนโมเดล (ใช้ค่าที่น้อยเพื่อเป็นตัวอย่าง)
    print("เริ่มการเทรนโมเดล...")
    trainer.train(
        dataset=train_dataset,
        output_dir="./results/llm_sft",
        num_train_epochs=1,
        per_device_train_batch_size=4,
        learning_rate=5e-5,
        weight_decay=0.01,
        save_steps=500,
        eval_dataset=eval_dataset
    )
    
    # บันทึกโมเดล
    print("บันทึกโมเดล...")
    trainer.save_model("./models/llm_sft_model")
    
    # ทดสอบการสร้างข้อความ
    prompt = "To be or not to be,"
    print(f"ทดสอบการสร้างข้อความจากโมเดลที่เทรนแล้ว โดยใช้ prompt: '{prompt}'")
    generated_texts = trainer.generate(prompt, max_length=50, num_return_sequences=3)
    
    for i, text in enumerate(generated_texts):
        print(f"ข้อความที่สร้าง #{i+1}: {text}")
    
    print("การทดสอบเสร็จสิ้น")

if __name__ == "__main__":
    main() 