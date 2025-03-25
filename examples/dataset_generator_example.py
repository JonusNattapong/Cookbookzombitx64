#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ตัวอย่างการใช้งานโมดูล DatasetGenerator สำหรับการสร้างชุดข้อมูล
"""

import os
import torch
from dotenv import load_dotenv
from cookbookzombitx64.data.dataset_generator import TextDatasetGenerator, ImageDatasetGenerator

# โหลดค่าตัวแปรสภาพแวดล้อม
load_dotenv()

def text_dataset_example():
    """ตัวอย่างการสร้างชุดข้อมูลข้อความ"""
    
    print("=== ตัวอย่างการสร้างชุดข้อมูลข้อความ ===")
    
    # สร้างออบเจ็กต์ TextDatasetGenerator
    text_generator = TextDatasetGenerator(seed=42)
    
    # ตัวอย่างข้อมูลสำหรับการสร้างชุดข้อมูลแบบ instruction tuning
    instructions = [
        {
            "instruction": "แปลข้อความนี้เป็นภาษาอังกฤษ: 'สวัสดีตอนเช้า ฉันชอบกินอาหารไทย'",
            "response": "Translate this text to English: 'Good morning, I like to eat Thai food.'"
        },
        {
            "instruction": "สรุปข้อความต่อไปนี้: 'ปัญญาประดิษฐ์ (AI) คือความสามารถของเครื่องจักรในการแสดงความฉลาดแบบมนุษย์ เช่น การเรียนรู้ การให้เหตุผล และการตัดสินใจ ในปัจจุบัน AI มีบทบาทสำคัญในหลายอุตสาหกรรม เช่น การแพทย์ การเงิน และการขนส่ง'",
            "response": "ปัญญาประดิษฐ์คือเทคโนโลยีที่ทำให้เครื่องจักรมีความฉลาดคล้ายมนุษย์ มีความสามารถในการเรียนรู้และตัดสินใจ ปัจจุบันใช้งานในหลายอุตสาหกรรมสำคัญ"
        },
        {
            "instruction": "ช่วยแต่งเพลงสั้นๆ เกี่ยวกับความสุขในวันฝนตก",
            "response": "สายฝนโปรยปราย\nหัวใจเต้นตามจังหวะหยดน้ำ\nความสุขล้นทะลัก เมื่อสัมผัสสายลม\nฉันนั่งมองฟ้า ร้องเพลงคลอเบาๆ\nวันนี้ฉันมีความสุข แม้ฟ้าจะร้องไห้"
        },
    ]
    
    # สร้างชุดข้อมูลแบบ instruction tuning
    instruction_dataset = text_generator.create_instruction_dataset(instructions)
    print(f"สร้างชุดข้อมูลแบบ instruction จำนวน {len(instruction_dataset)} ตัวอย่าง")
    print(f"ตัวอย่างข้อมูล: {instruction_dataset[0]}")
    
    # ตัวอย่างข้อมูลสำหรับการสร้างชุดข้อมูลแบบ preference
    prompts = [
        "ให้คำแนะนำวิธีการลดความเครียดในการทำงาน",
        "อธิบายวิธีการปลูกต้นไม้ในพื้นที่จำกัด",
        "เขียนโค้ด Python สำหรับการแสดงตารางสูตรคูณ"
    ]
    
    chosen_responses = [
        "การลดความเครียดในการทำงานสามารถทำได้หลายวิธี เช่น การจัดตารางเวลาทำงานให้เหมาะสม การพักสั้นๆ ระหว่างวัน การออกกำลังกายสม่ำเสมอ การฝึกสมาธิหรือโยคะ และการแบ่งเวลาให้กับงานอดิเรกที่ชอบ",
        "วิธีปลูกต้นไม้ในพื้นที่จำกัด: 1) เลือกพันธุ์ไม้ขนาดเล็กหรือไม้ที่เหมาะกับการปลูกในที่แคบ 2) ใช้ระบบปลูกแนวตั้ง เช่น สวนแนวตั้ง หรือชั้นวางต้นไม้หลายชั้น 3) ปลูกในภาชนะแขวนเพื่อประหยัดพื้นที่ 4) เลือกดินและปุ๋ยที่มีคุณภาพดี 5) จัดระบบน้ำและแสงให้เหมาะสม",
        "```python\ndef multiplication_table(n):\n    for i in range(1, n+1):\n        for j in range(1, n+1):\n            print(f\"{i} x {j} = {i*j}\")\n        print(\"-------------------\")\n\n# แสดงตารางสูตรคูณแม่ 1-12\nmultiplication_table(12)\n```"
    ]
    
    rejected_responses = [
        "ก็แค่ไม่ต้องคิดมาก ทำงานไปเรื่อยๆ แล้วก็พักบ้าง",
        "ปลูกต้นไม้ในกระถางเล็กๆ แล้วรดน้ำทุกวัน",
        "เขียนโค้ดก็ใช้ loop 2 ชั้น ลูปแรกวนแม่ ลูปสองวนตัวคูณ แล้วก็ print ออกมา"
    ]
    
    # สร้างชุดข้อมูลแบบ preference
    preference_dataset = text_generator.create_preference_dataset(prompts, chosen_responses, rejected_responses)
    print(f"\nสร้างชุดข้อมูลแบบ preference จำนวน {len(preference_dataset)} ตัวอย่าง")
    print(f"ตัวอย่างข้อมูล:")
    print(f"  - Prompt: {preference_dataset[0]['prompt']}")
    print(f"  - Chosen: {preference_dataset[0]['chosen'][:100]}...")
    print(f"  - Rejected: {preference_dataset[0]['rejected'][:100]}...")
    
    # แบ่งชุดข้อมูลเป็นส่วนฝึก ส่วนตรวจสอบ และส่วนทดสอบ
    splits = text_generator.train_val_test_split(preference_dataset)
    print(f"\nแบ่งชุดข้อมูล:")
    print(f"  - Train: {len(splits['train'])} ตัวอย่าง")
    print(f"  - Validation: {len(splits['validation'])} ตัวอย่าง")
    print(f"  - Test: {len(splits['test'])} ตัวอย่าง")
    
    # บันทึกชุดข้อมูล
    output_dir = "./datasets/preference_dataset"
    text_generator.save_dataset(splits, output_dir=output_dir, format="json")
    print(f"\nบันทึกชุดข้อมูลที่: {output_dir}")

def image_dataset_example():
    """ตัวอย่างการสร้างชุดข้อมูลภาพ (จำลองข้อมูล)"""
    
    print("\n=== ตัวอย่างการสร้างชุดข้อมูลภาพ (จำลอง) ===")
    
    # สร้างออบเจ็กต์ ImageDatasetGenerator
    image_generator = ImageDatasetGenerator(seed=42)
    
    # จำลองข้อมูลพาธภาพ
    image_paths = [
        "/path/to/image1.jpg",
        "/path/to/image2.jpg",
        "/path/to/image3.jpg",
        "/path/to/image4.jpg",
        "/path/to/image5.jpg"
    ]
    
    # จำลองข้อมูลคำอธิบายภาพ
    captions = [
        "ภาพพระอาทิตย์ตกที่ชายหาด แสงสีส้มสะท้อนบนผิวน้ำ",
        "ภาพแมวสีดำกำลังนอนบนโซฟาสีแดง",
        "ภาพป่าไม้ในฤดูใบไม้ร่วง ใบไม้สีเหลืองและแดงร่วงหล่นบนพื้น",
        "ภาพตึกสูงในเมืองใหญ่ยามค่ำคืน มีแสงไฟสว่างไสว",
        "ภาพอาหารจานพิเศษบนโต๊ะอาหาร มีการจัดวางอย่างสวยงาม"
    ]
    
    # สร้างชุดข้อมูลสำหรับการสร้างคำอธิบายภาพ
    captioning_dataset = image_generator.create_image_captioning_dataset(image_paths, captions)
    print(f"สร้างชุดข้อมูลสำหรับการสร้างคำอธิบายภาพ จำนวน {len(captioning_dataset)} ตัวอย่าง")
    print(f"ตัวอย่างข้อมูล: {captioning_dataset[0]}")
    
    # จำลองข้อมูลสำหรับการจำแนกภาพ
    labeled_images = {
        "แมว": [
            {"file_name": "cat1.jpg", "file_path": "/path/to/cats/cat1.jpg"},
            {"file_name": "cat2.jpg", "file_path": "/path/to/cats/cat2.jpg"}
        ],
        "สุนัข": [
            {"file_name": "dog1.jpg", "file_path": "/path/to/dogs/dog1.jpg"},
            {"file_name": "dog2.jpg", "file_path": "/path/to/dogs/dog2.jpg"}
        ],
        "นก": [
            {"file_name": "bird1.jpg", "file_path": "/path/to/birds/bird1.jpg"}
        ]
    }
    
    # สร้างชุดข้อมูลสำหรับการจำแนกภาพ
    classification_dataset = image_generator.create_image_classification_dataset(labeled_images)
    print(f"\nสร้างชุดข้อมูลสำหรับการจำแนกภาพ จำนวน {len(classification_dataset)} ตัวอย่าง")
    print(f"ตัวอย่างข้อมูล: {classification_dataset[0]}")
    
    # แบ่งชุดข้อมูลเป็นส่วนฝึก ส่วนตรวจสอบ และส่วนทดสอบ
    splits = image_generator.train_val_test_split(classification_dataset)
    print(f"\nแบ่งชุดข้อมูล:")
    print(f"  - Train: {len(splits['train'])} ตัวอย่าง")
    print(f"  - Validation: {len(splits['validation'])} ตัวอย่าง")
    print(f"  - Test: {len(splits['test'])} ตัวอย่าง")

def main():
    """ฟังก์ชันหลักสำหรับตัวอย่างการใช้งาน"""
    
    # ตัวอย่างการสร้างชุดข้อมูลข้อความ
    text_dataset_example()
    
    # ตัวอย่างการสร้างชุดข้อมูลภาพ (จำลองข้อมูล)
    image_dataset_example()
    
    print("\nการทดสอบโมดูล DatasetGenerator เสร็จสิ้น")

if __name__ == "__main__":
    main() 