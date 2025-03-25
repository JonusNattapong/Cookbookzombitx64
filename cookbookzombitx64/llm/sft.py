"""
โมดูลสำหรับการปรับแต่งโมเดลด้วยเทคนิค Supervised Fine-Tuning (SFT)
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

class LLMSFTTrainer:
    """คลาสสำหรับการปรับแต่งโมเดลด้วยเทคนิค Supervised Fine-Tuning"""
    
    def __init__(self, model_name, tokenizer_name=None, device=None):
        """
        กำหนดค่าเริ่มต้นสำหรับการปรับแต่งโมเดล
        
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
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name).to(self.device)
        
    def prepare_dataset(self, dataset, text_column="text", max_length=512):
        """
        เตรียมชุดข้อมูลสำหรับการเทรน
        
        Args:
            dataset: ชุดข้อมูลสำหรับการเทรน
            text_column (str): ชื่อคอลัมน์ที่เก็บข้อความ
            max_length (int): ความยาวสูงสุดของข้อความ
            
        Returns:
            dataset: ชุดข้อมูลที่เตรียมพร้อมแล้ว
        """
        def tokenize_function(examples):
            return self.tokenizer(examples[text_column], padding="max_length", truncation=True, max_length=max_length)
        
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        tokenized_dataset = tokenized_dataset.remove_columns([col for col in dataset.column_names if col != text_column])
        tokenized_dataset = tokenized_dataset.rename_column(text_column, "input_ids")
        tokenized_dataset.set_format("torch")
        
        return tokenized_dataset
    
    def train(self, dataset, output_dir="./results", num_train_epochs=3, per_device_train_batch_size=8, 
              learning_rate=5e-5, weight_decay=0.01, save_steps=1000, eval_dataset=None):
        """
        เทรนโมเดลด้วยเทคนิค Supervised Fine-Tuning
        
        Args:
            dataset: ชุดข้อมูลที่เตรียมพร้อมแล้ว
            output_dir (str): ไดเรกทอรีสำหรับบันทึกผลลัพธ์
            num_train_epochs (int): จำนวนรอบการเทรน
            per_device_train_batch_size (int): ขนาดแบตช์ต่ออุปกรณ์
            learning_rate (float): อัตราการเรียนรู้
            weight_decay (float): ค่าการลดน้ำหนัก
            save_steps (int): จำนวนขั้นตอนก่อนที่จะบันทึกโมเดล
            eval_dataset: ชุดข้อมูลสำหรับการประเมินผล (ถ้ามี)
            
        Returns:
            trainer: ออบเจ็กต์ Trainer หลังการเทรน
        """
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            save_steps=save_steps,
            logging_dir=f"{output_dir}/logs",
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            eval_dataset=eval_dataset,
        )
        
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