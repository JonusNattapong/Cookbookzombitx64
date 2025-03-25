"""
โมดูลสำหรับการปรับแต่งโมเดล Vision Language Model (VLM) สำหรับการตอบคำถามจากภาพ (Visual Question Answering)
"""

import os
import torch
import torch.nn.functional as F
from transformers import (
    ViltProcessor, 
    ViltForQuestionAnswering,
    Trainer, 
    TrainingArguments
)
from datasets import Dataset
from PIL import Image
from typing import List, Dict, Optional, Union, Tuple, Any

class VLMVQATrainer:
    """คลาสสำหรับการปรับแต่งโมเดล VLM สำหรับการตอบคำถามจากภาพ"""
    
    def __init__(
        self, 
        model_name: str = "dandelin/vilt-b32-finetuned-vqa",
        processor_name: Optional[str] = None,
        device: Optional[str] = None
    ):
        """
        กำหนดค่าเริ่มต้นสำหรับการปรับแต่งโมเดล VQA
        
        Args:
            model_name (str): ชื่อหรือพาธของโมเดล VQA
            processor_name (str, optional): ชื่อหรือพาธของ processor ถ้าไม่ระบุจะใช้ค่าเดียวกับ model_name
            device (str, optional): อุปกรณ์ที่ใช้ในการคำนวณ ('cuda', 'cpu') ถ้าไม่ระบุจะเลือกอัตโนมัติ
        """
        # กำหนดอุปกรณ์ที่ใช้ในการคำนวณ
        if device:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
        # โหลดโมเดลและ processor
        self.model = ViltForQuestionAnswering.from_pretrained(model_name).to(self.device)
        processor_name = processor_name or model_name
        self.processor = ViltProcessor.from_pretrained(processor_name)
        
        # เก็บแมปปิงระหว่าง ID และคำตอบ
        self.id2label = self.model.config.id2label if hasattr(self.model.config, 'id2label') else {}
        self.label2id = {v: k for k, v in self.id2label.items()} if self.id2label else {}
    
    def prepare_dataset(
        self, 
        image_paths: List[str], 
        questions: List[str], 
        answers: List[Union[str, List[str]]], 
        max_length: int = 64
    ) -> Dataset:
        """
        เตรียมชุดข้อมูลสำหรับการเทรน VQA
        
        Args:
            image_paths (List[str]): รายการของพาธไฟล์ภาพ
            questions (List[str]): รายการของคำถาม
            answers (List[Union[str, List[str]]]): รายการของคำตอบ (อาจเป็นคำตอบเดียวหรือหลายคำตอบ)
            max_length (int): ความยาวสูงสุดของคำถาม
            
        Returns:
            Dataset: ชุดข้อมูลที่เตรียมพร้อมแล้ว
        """
        if len(image_paths) != len(questions) or len(image_paths) != len(answers):
            raise ValueError("จำนวนของ image_paths, questions และ answers ต้องเท่ากัน")
            
        # แปลงคำตอบให้เป็นรูปแบบที่สอดคล้องกัน (ทุกคำตอบเป็นลิสต์)
        normalized_answers = []
        for ans in answers:
            if isinstance(ans, str):
                normalized_answers.append([ans])
            else:
                normalized_answers.append(ans)
                
        # สร้างชุดข้อมูล
        data = {
            "image_path": image_paths,
            "question": questions,
            "answer": normalized_answers
        }
        
        dataset = Dataset.from_dict(data)
        
        # สร้างหรืออัปเดตแมปปิง label
        self._update_label_mapping(normalized_answers)
        
        # เตรียมข้อมูลสำหรับการเทรน
        def preprocess_data(examples):
            # โหลดภาพ
            images = [Image.open(image_path).convert("RGB") for image_path in examples["image_path"]]
            
            # แปลงข้อมูลด้วย processor
            encodings = [
                self.processor(image=image, text=question, padding="max_length", truncation=True, max_length=max_length, return_tensors="pt")
                for image, question in zip(images, examples["question"])
            ]
            
            # รวมข้อมูลจากทุกตัวอย่าง
            batch_encodings = {
                'pixel_values': torch.stack([encoding['pixel_values'][0] for encoding in encodings]),
                'input_ids': torch.stack([encoding['input_ids'][0] for encoding in encodings]),
                'attention_mask': torch.stack([encoding['attention_mask'][0] for encoding in encodings]),
                'token_type_ids': torch.stack([encoding['token_type_ids'][0] for encoding in encodings])
            }
            
            # แปลงคำตอบเป็น label_id
            labels = []
            for ans_list in examples["answer"]:
                # เลือกคำตอบแรกเป็นหลัก (ถ้ามีหลายคำตอบ)
                ans = ans_list[0]
                if ans in self.label2id:
                    labels.append(self.label2id[ans])
                else:
                    # ถ้าไม่มีในแมปปิง ใช้ค่า 0 หรือเพิ่มในแมปปิง
                    self.id2label[len(self.id2label)] = ans
                    self.label2id[ans] = len(self.label2id) - 1
                    labels.append(self.label2id[ans])
            
            batch_encodings['labels'] = torch.tensor(labels)
            
            return batch_encodings
        
        processed_dataset = dataset.map(preprocess_data, batched=True, remove_columns=["image_path", "question", "answer"])
        
        return processed_dataset
    
    def _update_label_mapping(self, all_answers: List[List[str]]):
        """
        อัปเดตแมปปิงระหว่าง ID และคำตอบ
        
        Args:
            all_answers (List[List[str]]): รายการของรายการคำตอบทั้งหมด
        """
        # รวมคำตอบทั้งหมดและลบคำตอบซ้ำ
        unique_answers = set()
        for ans_list in all_answers:
            for ans in ans_list:
                unique_answers.add(ans)
        
        # อัปเดตแมปปิง
        for ans in unique_answers:
            if ans not in self.label2id:
                new_id = len(self.id2label)
                self.id2label[new_id] = ans
                self.label2id[ans] = new_id
    
    def train(
        self, 
        dataset: Dataset, 
        output_dir: str = "./results", 
        num_train_epochs: int = 3, 
        per_device_train_batch_size: int = 16, 
        learning_rate: float = 5e-5, 
        weight_decay: float = 0.01, 
        save_steps: int = 1000, 
        eval_dataset: Optional[Dataset] = None
    ):
        """
        เทรนโมเดล VQA
        
        Args:
            dataset (Dataset): ชุดข้อมูลที่เตรียมพร้อมแล้ว
            output_dir (str): ไดเรกทอรีสำหรับบันทึกผลลัพธ์
            num_train_epochs (int): จำนวนรอบการเทรน
            per_device_train_batch_size (int): ขนาดแบตช์ต่ออุปกรณ์
            learning_rate (float): อัตราการเรียนรู้
            weight_decay (float): ค่าการลดน้ำหนัก
            save_steps (int): จำนวนขั้นตอนก่อนที่จะบันทึกโมเดล
            eval_dataset (Dataset, optional): ชุดข้อมูลสำหรับการประเมินผล
        """
        # อัปเดตแมปปิงใน model config
        self.model.config.id2label = self.id2label
        self.model.config.label2id = self.label2id
        
        # กำหนดค่าพารามิเตอร์สำหรับการเทรน
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            save_steps=save_steps,
            logging_dir=f"{output_dir}/logs",
        )
        
        # สร้าง trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            eval_dataset=eval_dataset,
        )
        
        # เริ่มการเทรน
        trainer.train()
        
        return trainer
    
    def save_model(self, path: str):
        """
        บันทึกโมเดลและ processor
        
        Args:
            path (str): พาธที่จะบันทึกโมเดล
        """
        # สร้างไดเรกทอรีถ้ายังไม่มี
        os.makedirs(path, exist_ok=True)
        
        self.model.save_pretrained(path)
        self.processor.save_pretrained(path)
        
        # บันทึกแมปปิง label
        with open(os.path.join(path, "id2label.txt"), "w") as f:
            for id_, label in self.id2label.items():
                f.write(f"{id_}\t{label}\n")
    
    def answer_question(self, image_path: str, question: str) -> Dict[str, Any]:
        """
        ตอบคำถามจากภาพ
        
        Args:
            image_path (str): พาธของไฟล์ภาพ
            question (str): คำถาม
            
        Returns:
            Dict[str, Any]: คำตอบและความน่าจะเป็น
        """
        # โหลดภาพ
        image = Image.open(image_path).convert("RGB")
        
        # แปลงข้อมูลด้วย processor
        encoding = self.processor(image=image, text=question, return_tensors="pt")
        
        # ย้ายข้อมูลไปยังอุปกรณ์
        for key in encoding:
            encoding[key] = encoding[key].to(self.device)
        
        # ตอบคำถาม
        with torch.no_grad():
            outputs = self.model(**encoding)
            logits = outputs.logits
            
            # หาคำตอบที่มีความน่าจะเป็นสูงสุด
            probs = F.softmax(logits, dim=1)[0]
            top_k_values, top_k_indices = torch.topk(probs, k=5)
            
            # แปลงกลับเป็นคำตอบ
            results = []
            for i, (value, idx) in enumerate(zip(top_k_values.tolist(), top_k_indices.tolist())):
                answer = self.id2label.get(idx, "unknown")
                results.append({
                    "answer": answer,
                    "probability": value,
                    "rank": i + 1
                })
        
        return {
            "question": question,
            "top_answers": results,
            "best_answer": results[0]["answer"] if results else "unknown"
        } 