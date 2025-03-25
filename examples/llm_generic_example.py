#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ตัวอย่างการใช้งานโมดูล LLM Generic สำหรับการปรับแต่งโมเดลภาษาพื้นฐานทั่วไป
"""

import os
import torch
import logging
from dotenv import load_dotenv
from datasets import Dataset
from cookbookzombitx64.llm.generic import LLMGenericTrainer

# ตั้งค่า logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# โหลดค่าตัวแปรสภาพแวดล้อม
load_dotenv()

def main():
    """ฟังก์ชันหลักสำหรับตัวอย่างการใช้งาน"""
    
    # ตั้งค่าโมเดลที่จะใช้
    model_name = "gpt2"  # ใช้โมเดล GPT-2 ขนาดเล็กเพื่อเป็นตัวอย่าง
    
    print(f"เริ่มต้นการใช้งาน LLM Generic Trainer ด้วยโมเดล {model_name}")
    
    # สร้างออบเจ็กต์ LLMGenericTrainer
    trainer = LLMGenericTrainer(
        model_name,
        model_kwargs={"low_cpu_mem_usage": True},
        tokenizer_kwargs={"use_fast": True}
    )
    
    print(f"โมเดลทำงานบนอุปกรณ์: {trainer.device}")
    
    # สร้างชุดข้อมูลตัวอย่างสำหรับการฝึกแบบ language modeling
    texts = [
        "ปัญญาประดิษฐ์ (AI) คือความสามารถของเครื่องจักรในการแสดงความฉลาดแบบมนุษย์ ในปัจจุบัน AI มีบทบาทสำคัญในหลายอุตสาหกรรม",
        "การเรียนรู้เชิงลึก (Deep Learning) เป็นเทคนิคการเรียนรู้ของเครื่องที่ใช้โครงข่ายประสาทเทียมหลายชั้น",
        "ภาษาไทยเป็นภาษาที่มีความซับซ้อนทั้งในด้านไวยากรณ์และคำศัพท์ การประมวลผลภาษาธรรมชาติสำหรับภาษาไทยจึงเป็นความท้าทาย",
        "การปรับแต่งโมเดลภาษาขนาดใหญ่สามารถทำได้หลายวิธี เช่น การปรับแต่งแบบมีผู้สอน (Supervised Fine-tuning) และการเรียนรู้จากคำแนะนำของมนุษย์",
        "ข้อมูลฝึกสอนที่มีคุณภาพเป็นปัจจัยสำคัญในความสำเร็จของโมเดลเอไอ ข้อมูลที่หลากหลายและครอบคลุมจะช่วยให้โมเดลมีประสิทธิภาพดีขึ้น",
        "การประยุกต์ใช้ปัญญาประดิษฐ์ในชีวิตประจำวันมีหลายรูปแบบ เช่น ระบบแนะนำสินค้า การแปลภาษา และการวิเคราะห์ภาพถ่าย",
        "การพัฒนาโมเดลเอไอที่มีความรับผิดชอบต้องคำนึงถึงความเป็นธรรม ความโปร่งใส และความเป็นส่วนตัวของผู้ใช้",
        "เทคโนโลยีการประมวลผลภาษาธรรมชาติช่วยให้คอมพิวเตอร์เข้าใจและสร้างภาษามนุษย์ได้ ทำให้เกิดแอปพลิเคชันต่างๆ เช่น ระบบช่วยเขียน และโปรแกรมสนทนาอัตโนมัติ",
        "การปรับแต่งโมเดลภาษาทำให้โมเดลสามารถทำงานได้ดีในงานเฉพาะทาง เช่น การวิเคราะห์ความรู้สึก การสรุปความ และการตอบคำถาม",
        "ฐานข้อมูลความรู้ที่ครอบคลุมและถูกต้องเป็นพื้นฐานสำคัญของระบบปัญญาประดิษฐ์ที่ให้คำตอบที่น่าเชื่อถือ"
    ]
    
    # สร้างชุดข้อมูลจากข้อความ
    dataset = Dataset.from_dict({"text": texts})
    print(f"สร้างชุดข้อมูลมี {len(dataset)} ตัวอย่าง")
    
    # เตรียมชุดข้อมูลสำหรับการเทรน
    print("เตรียมชุดข้อมูลสำหรับการเทรน...")
    prepared_dataset = trainer.prepare_dataset(dataset, text_column="text", max_length=128)
    
    # แบ่งชุดข้อมูลเป็นส่วนฝึกและส่วนตรวจสอบ
    prepared_dataset = prepared_dataset.train_test_split(test_size=0.2, seed=42)
    train_dataset = prepared_dataset["train"]
    eval_dataset = prepared_dataset["test"]
    print(f"แบ่งชุดข้อมูลเป็น: ฝึก {len(train_dataset)} ตัวอย่าง, ตรวจสอบ {len(eval_dataset)} ตัวอย่าง")
    
    # สร้างโฟลเดอร์เก็บผลลัพธ์
    output_dir = "./results/llm_generic"
    os.makedirs(output_dir, exist_ok=True)
    
    # เทรนโมเดล (ใช้ค่าที่น้อยเพื่อเป็นตัวอย่าง)
    print("เริ่มการเทรนโมเดล...")
    trainer.train(
        dataset=train_dataset,
        eval_dataset=eval_dataset,
        output_dir=output_dir,
        num_train_epochs=2,
        per_device_train_batch_size=2,
        learning_rate=5e-5,
        weight_decay=0.01,
        save_steps=10,
        logging_steps=5,
        evaluation_strategy="steps",
        eval_steps=10,
        fp16=torch.cuda.is_available()
    )
    
    # บันทึกโมเดล
    print("บันทึกโมเดล...")
    model_output_dir = os.path.join(output_dir, "final_model")
    trainer.save_model(model_output_dir)
    print(f"บันทึกโมเดลที่: {model_output_dir}")
    
    # ทดสอบการสร้างข้อความ
    test_prompts = [
        "ปัญญาประดิษฐ์คือ",
        "การพัฒนาโมเดลภาษามีขั้นตอนดังนี้",
        "ประโยชน์ของเทคโนโลยี AI ในชีวิตประจำวัน"
    ]
    
    print("\nทดสอบการสร้างข้อความจากโมเดลที่เทรนแล้ว:")
    for prompt in test_prompts:
        print(f"\nPrompt: {prompt}")
        generated_texts = trainer.generate(
            prompt=prompt, 
            max_length=150,
            temperature=0.7,
            top_p=0.95,
            repetition_penalty=1.2
        )
        print(f"คำตอบ: {generated_texts[0]}")
    
    print("\nการทดสอบเสร็จสิ้น")

if __name__ == "__main__":
    main() 