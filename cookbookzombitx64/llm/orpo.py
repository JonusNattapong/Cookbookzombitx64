"""
โมดูลสำหรับการปรับแต่งโมเดลด้วยเทคนิค ORPO (Optimized Rank Preference Optimization)
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import Dataset

class LLMORPOTrainer:
    """คลาสสำหรับการปรับแต่งโมเดลด้วยเทคนิค ORPO"""
    
    def __init__(self, model_name, tokenizer_name=None, device=None, beta=0.1):
        """
        กำหนดค่าเริ่มต้นสำหรับการปรับแต่งโมเดล
        
        Args:
            model_name (str): ชื่อหรือพาธของโมเดลที่จะทำการปรับแต่ง
            tokenizer_name (str, optional): ชื่อหรือพาธของ tokenizer ถ้าไม่ระบุจะใช้ค่าเดียวกับ model_name
            device (str, optional): อุปกรณ์ที่ใช้ในการคำนวณ ('cuda', 'cpu') ถ้าไม่ระบุจะเลือกอัตโนมัติ
            beta (float): พารามิเตอร์ beta สำหรับ ORPO (ค่าเริ่มต้น 0.1)
        """
        self.model_name = model_name
        self.tokenizer_name = tokenizer_name if tokenizer_name else model_name
        self.beta = beta
        
        # กำหนดอุปกรณ์ที่ใช้ในการคำนวณ
        if device:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
        # โหลดโมเดลและ tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name).to(self.device)
        
    def prepare_dataset(self, prompts, chosen_responses, rejected_responses, max_length=512):
        """
        เตรียมชุดข้อมูลสำหรับการเทรน ORPO
        
        Args:
            prompts (list): รายการของ prompt
            chosen_responses (list): รายการของคำตอบที่ถูกเลือก (preferred)
            rejected_responses (list): รายการของคำตอบที่ถูกปฏิเสธ (less preferred)
            max_length (int): ความยาวสูงสุดของข้อความ
            
        Returns:
            dataset: ชุดข้อมูลที่เตรียมพร้อมแล้ว
        """
        if len(prompts) != len(chosen_responses) or len(prompts) != len(rejected_responses):
            raise ValueError("จำนวนของ prompts, chosen_responses และ rejected_responses ต้องเท่ากัน")
            
        data = {
            "prompt": prompts,
            "chosen": chosen_responses,
            "rejected": rejected_responses
        }
        
        dataset = Dataset.from_dict(data)
        
        # เตรียมข้อมูลสำหรับการเทรน
        def tokenize_function(examples):
            chosen_inputs = [prompt + chosen for prompt, chosen in zip(examples["prompt"], examples["chosen"])]
            rejected_inputs = [prompt + rejected for prompt, rejected in zip(examples["prompt"], examples["rejected"])]
            
            chosen_tokens = self.tokenizer(chosen_inputs, padding="max_length", truncation=True, max_length=max_length, return_tensors="pt")
            rejected_tokens = self.tokenizer(rejected_inputs, padding="max_length", truncation=True, max_length=max_length, return_tensors="pt")
            
            # สร้าง prompt mask สำหรับแยกส่วน prompt ออกจากคำตอบในการคำนวณ loss
            prompt_tokens = self.tokenizer(examples["prompt"], padding="max_length", truncation=True, max_length=max_length, return_tensors="pt")
            prompt_mask = []
            
            for i in range(len(prompt_tokens["input_ids"])):
                # นับความยาวของ prompt (ไม่รวม padding)
                prompt_len = (prompt_tokens["attention_mask"][i] == 1).sum().item()
                # สร้าง mask โดยกำหนดให้ส่วนของ prompt เป็น 0 และส่วนของคำตอบเป็น 1
                mask = [0] * prompt_len + [1] * (max_length - prompt_len)
                mask = mask[:max_length]  # ตัดให้มีความยาวไม่เกิน max_length
                prompt_mask.append(mask)
            
            return {
                "chosen_input_ids": chosen_tokens["input_ids"],
                "chosen_attention_mask": chosen_tokens["attention_mask"],
                "rejected_input_ids": rejected_tokens["input_ids"],
                "rejected_attention_mask": rejected_tokens["attention_mask"],
                "prompt_mask": torch.tensor(prompt_mask)
            }
        
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        tokenized_dataset.set_format(type="torch", columns=[
            "chosen_input_ids", "chosen_attention_mask", 
            "rejected_input_ids", "rejected_attention_mask",
            "prompt_mask"
        ])
        
        return tokenized_dataset
    
    def orpo_loss(self, chosen_logits, rejected_logits, prompt_mask):
        """
        คำนวณ ORPO loss
        
        Args:
            chosen_logits: logits จากคำตอบที่ถูกเลือก
            rejected_logits: logits จากคำตอบที่ถูกปฏิเสธ
            prompt_mask: mask สำหรับแยกส่วน prompt ออกจากคำตอบ
            
        Returns:
            loss: ค่า loss ที่คำนวณได้
        """
        # คำนวณ log probabilities
        chosen_log_probs = F.log_softmax(chosen_logits, dim=-1)
        rejected_log_probs = F.log_softmax(rejected_logits, dim=-1)
        
        # ใช้เฉพาะส่วนของคำตอบในการคำนวณ loss
        chosen_log_probs = chosen_log_probs.masked_select(prompt_mask.unsqueeze(-1).bool())
        rejected_log_probs = rejected_log_probs.masked_select(prompt_mask.unsqueeze(-1).bool())
        
        # คำนวณ ORPO loss
        chosen_rewards = chosen_log_probs.sum()
        rejected_rewards = rejected_log_probs.sum()
        
        # คำนวณ margin loss: max(0, β - (chosen_rewards - rejected_rewards))
        loss = torch.max(torch.tensor(0.0).to(self.device), self.beta - (chosen_rewards - rejected_rewards))
        
        return loss
    
    def train(self, dataset, output_dir="./results", num_train_epochs=3, per_device_train_batch_size=4, 
              learning_rate=1e-5, weight_decay=0.01, save_steps=1000, eval_dataset=None):
        """
        เทรนโมเดลด้วยเทคนิค ORPO
        
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
            chosen_input_ids = inputs["chosen_input_ids"].to(self.device)
            chosen_attention_mask = inputs["chosen_attention_mask"].to(self.device)
            rejected_input_ids = inputs["rejected_input_ids"].to(self.device)
            rejected_attention_mask = inputs["rejected_attention_mask"].to(self.device)
            prompt_mask = inputs["prompt_mask"].to(self.device)
            
            # คำนวณ logits สำหรับคำตอบที่ถูกเลือกและถูกปฏิเสธ
            chosen_outputs = model(input_ids=chosen_input_ids, attention_mask=chosen_attention_mask)
            rejected_outputs = model(input_ids=rejected_input_ids, attention_mask=rejected_attention_mask)
            
            # คำนวณ loss
            loss = self.orpo_loss(chosen_outputs.logits, rejected_outputs.logits, prompt_mask)
            
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
        
    def generate(self, prompt, max_length=100, num_return_sequences=1, **kwargs):
        """
        สร้างข้อความจากโมเดล
        
        Args:
            prompt (str): ข้อความเริ่มต้น
            max_length (int): ความยาวสูงสุดของข้อความที่สร้าง
            num_return_sequences (int): จำนวนข้อความที่จะสร้าง
            **kwargs: พารามิเตอร์เพิ่มเติมสำหรับการสร้าง
            
        Returns:
            list: รายการของข้อความที่สร้างขึ้น
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        outputs = self.model.generate(
            inputs.input_ids,
            max_length=max_length,
            num_return_sequences=num_return_sequences,
            **kwargs
        )
        
        return [self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs] 