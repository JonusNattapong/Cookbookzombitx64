"""
โมดูลสำหรับการปรับแต่งโมเดลด้วยเทคนิค Reward Modeling (RM) สำหรับ RLHF
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import Dataset
from typing import List, Dict, Union, Optional, Tuple

class LLMRewardTrainer:
    """คลาสสำหรับการปรับแต่งโมเดล Reward Model สำหรับ RLHF"""
    
    def __init__(self, model_name, tokenizer_name=None, device=None):
        """
        กำหนดค่าเริ่มต้นสำหรับการปรับแต่งโมเดล Reward
        
        Args:
            model_name (str): ชื่อหรือพาธของโมเดลที่จะทำการปรับแต่ง
            tokenizer_name (str, optional): ชื่อหรือพาธของ tokenizer ถ้าไม่ระบุจะใช้ค่าเดียวกับ model_name
            device (str, optional): อุปกรณ์ที่ใช้ในการคำนวณ ('cuda', 'cpu') ถ้าไม่ระบุจะเลือกอัตโนมัติ
        """
        self.model_name = model_name
        self.tokenizer_name = tokenizer_name if tokenizer_name else model_name
        
        # กำหนดอุปกรณ์ที่ใช้ในการคำนวณ
        if device:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
        # โหลดโมเดลและ tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        # โหลดโมเดลสำหรับการจัดลำดับ/ให้คะแนน (1 output node)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=1).to(self.device)
        
    def prepare_dataset(self, preferred_texts: List[str], rejected_texts: List[str], max_length: int = 512) -> Dataset:
        """
        เตรียมชุดข้อมูลสำหรับการเทรน Reward Model
        
        Args:
            preferred_texts (List[str]): รายการของข้อความที่ต้องการ (preferred)
            rejected_texts (List[str]): รายการของข้อความที่ไม่ต้องการ (rejected)
            max_length (int): ความยาวสูงสุดของข้อความ
            
        Returns:
            Dataset: ชุดข้อมูลที่เตรียมพร้อมแล้ว
        """
        if len(preferred_texts) != len(rejected_texts):
            raise ValueError("จำนวนของ preferred_texts และ rejected_texts ต้องเท่ากัน")
            
        # สร้างชุดข้อมูล
        data = {
            "preferred": preferred_texts,
            "rejected": rejected_texts
        }
        
        dataset = Dataset.from_dict(data)
        
        # เตรียมข้อมูลสำหรับการเทรน
        def tokenize_function(examples):
            preferred_tokens = self.tokenizer(examples["preferred"], padding="max_length", truncation=True, max_length=max_length, return_tensors="pt")
            rejected_tokens = self.tokenizer(examples["rejected"], padding="max_length", truncation=True, max_length=max_length, return_tensors="pt")
            
            return {
                "preferred_input_ids": preferred_tokens["input_ids"],
                "preferred_attention_mask": preferred_tokens["attention_mask"],
                "rejected_input_ids": rejected_tokens["input_ids"],
                "rejected_attention_mask": rejected_tokens["attention_mask"],
            }
        
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        tokenized_dataset.set_format(type="torch", columns=[
            "preferred_input_ids", "preferred_attention_mask", 
            "rejected_input_ids", "rejected_attention_mask"
        ])
        
        return tokenized_dataset
    
    def reward_loss(self, preferred_rewards, rejected_rewards, margin=0.0):
        """
        คำนวณ loss สำหรับ Reward Model (Bradley-Terry loss)
        
        Args:
            preferred_rewards: คะแนนรางวัลของข้อความที่ต้องการ
            rejected_rewards: คะแนนรางวัลของข้อความที่ไม่ต้องการ
            margin (float): ค่า margin ระหว่างคะแนนรางวัล
            
        Returns:
            loss: ค่า loss ที่คำนวณได้
        """
        # คำนวณความน่าจะเป็นที่ preferred จะได้รับการเลือกมากกว่า rejected
        # P(preferred > rejected) = sigmoid(preferred_reward - rejected_reward)
        logits = preferred_rewards - rejected_rewards
        loss = -F.logsigmoid(logits - margin).mean()
        
        return loss
    
    def train(self, dataset, output_dir="./results", num_train_epochs=3, per_device_train_batch_size=8, 
              learning_rate=1e-5, weight_decay=0.01, save_steps=1000, eval_dataset=None):
        """
        เทรนโมเดล Reward
        
        Args:
            dataset: ชุดข้อมูลที่เตรียมพร้อมแล้ว
            output_dir (str): ไดเรกทอรีสำหรับบันทึกผลลัพธ์
            num_train_epochs (int): จำนวนรอบการเทรน
            per_device_train_batch_size (int): ขนาดแบตช์ต่ออุปกรณ์
            learning_rate (float): อัตราการเรียนรู้
            weight_decay (float): ค่าการลดน้ำหนัก
            save_steps (int): จำนวนขั้นตอนก่อนที่จะบันทึกโมเดล
            eval_dataset: ชุดข้อมูลสำหรับการประเมินผล (ถ้ามี)
        """
        def compute_loss(model, inputs):
            # แยกข้อมูลออกจาก inputs
            preferred_input_ids = inputs["preferred_input_ids"].to(self.device)
            preferred_attention_mask = inputs["preferred_attention_mask"].to(self.device)
            rejected_input_ids = inputs["rejected_input_ids"].to(self.device)
            rejected_attention_mask = inputs["rejected_attention_mask"].to(self.device)
            
            # คำนวณคะแนนรางวัลสำหรับข้อความที่ต้องการและไม่ต้องการ
            preferred_outputs = model(input_ids=preferred_input_ids, attention_mask=preferred_attention_mask)
            rejected_outputs = model(input_ids=rejected_input_ids, attention_mask=rejected_attention_mask)
            
            preferred_rewards = preferred_outputs.logits
            rejected_rewards = rejected_outputs.logits
            
            # คำนวณ loss
            loss = self.reward_loss(preferred_rewards, rejected_rewards)
            
            return loss
        
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
        
        # สร้าง trainer ด้วยฟังก์ชัน compute_loss ที่กำหนดเอง
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            eval_dataset=eval_dataset,
            compute_loss=compute_loss
        )
        
        # เริ่มการเทรน
        trainer.train()
        
        return trainer
    
    def save_model(self, path):
        """
        บันทึกโมเดลและ tokenizer
        
        Args:
            path (str): พาธที่จะบันทึกโมเดล
        """
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        
    def get_reward(self, texts: Union[str, List[str]]) -> List[float]:
        """
        คำนวณคะแนนรางวัลสำหรับข้อความที่กำหนด
        
        Args:
            texts (Union[str, List[str]]): ข้อความหรือรายการของข้อความที่จะคำนวณคะแนน
            
        Returns:
            List[float]: รายการของคะแนนรางวัล
        """
        if isinstance(texts, str):
            texts = [texts]
            
        # แปลงข้อความเป็น input_ids และ attention_mask
        encoded_texts = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
        
        # คำนวณคะแนนรางวัล
        with torch.no_grad():
            outputs = self.model(**encoded_texts)
            rewards = outputs.logits.squeeze(-1).cpu().tolist()
            
        return rewards
    
    def compare_texts(self, text1: str, text2: str) -> Dict[str, Union[float, str]]:
        """
        เปรียบเทียบคะแนนรางวัลระหว่างสองข้อความ
        
        Args:
            text1 (str): ข้อความที่หนึ่ง
            text2 (str): ข้อความที่สอง
            
        Returns:
            Dict[str, Union[float, str]]: ผลการเปรียบเทียบ
        """
        rewards = self.get_reward([text1, text2])
        reward1, reward2 = rewards
        
        result = {
            "reward1": reward1,
            "reward2": reward2,
            "difference": reward1 - reward2,
            "preferred": "text1" if reward1 > reward2 else "text2"
        }
        
        return result 