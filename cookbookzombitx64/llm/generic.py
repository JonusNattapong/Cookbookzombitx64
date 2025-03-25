"""
โมดูลสำหรับการปรับแต่งโมเดลภาษาขั้นพื้นฐานทั่วไป (Generic LLM Finetuning)
"""

import os
import torch
import logging
from typing import Dict, List, Optional, Union, Any
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import Dataset, load_dataset

class LLMGenericTrainer:
    """คลาสสำหรับการปรับแต่งโมเดลภาษาขั้นพื้นฐาน"""
    
    def __init__(
        self,
        model_name: str,
        tokenizer_name: Optional[str] = None,
        device: Optional[str] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        tokenizer_kwargs: Optional[Dict[str, Any]] = None
    ):
        """
        กำหนดค่าเริ่มต้นสำหรับการปรับแต่งโมเดล
        
        Args:
            model_name (str): ชื่อหรือพาธของโมเดลที่จะทำการปรับแต่ง
            tokenizer_name (str, optional): ชื่อหรือพาธของ tokenizer ถ้าไม่ระบุจะใช้ค่าเดียวกับ model_name
            device (str, optional): อุปกรณ์ที่ใช้ในการคำนวณ ('cuda', 'cpu') ถ้าไม่ระบุจะเลือกอัตโนมัติ
            model_kwargs (Dict[str, Any], optional): พารามิเตอร์เพิ่มเติมสำหรับการโหลดโมเดล
            tokenizer_kwargs (Dict[str, Any], optional): พารามิเตอร์เพิ่มเติมสำหรับการโหลด tokenizer
        """
        self.model_name = model_name
        self.tokenizer_name = tokenizer_name if tokenizer_name else model_name
        
        # กำหนดค่าเริ่มต้นถ้าไม่ได้ระบุ
        if model_kwargs is None:
            model_kwargs = {}
        if tokenizer_kwargs is None:
            tokenizer_kwargs = {}
            
        # ตั้งค่า default สำหรับ tokenizer
        if 'padding_side' not in tokenizer_kwargs:
            tokenizer_kwargs['padding_side'] = 'right'
        
        # กำหนดอุปกรณ์ที่ใช้ในการคำนวณ
        if device:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
        # โหลด tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_name, 
            **tokenizer_kwargs
        )
        
        # ตรวจสอบว่ามี pad_token หรือไม่
        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                self.tokenizer.pad_token = self.tokenizer.eos_token = "</s>"
                
        # โหลดโมเดล
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map=self.device if self.device != "cpu" else None,
                **model_kwargs
            )
        except Exception as e:
            logging.error(f"ไม่สามารถโหลดโมเดลได้: {str(e)}")
            raise
            
        logging.info(f"โหลดโมเดล {self.model_name} เรียบร้อยแล้ว อุปกรณ์: {self.device}")
        
    def prepare_dataset(
        self,
        dataset: Union[Dataset, str],
        text_column: str = "text",
        max_length: int = 512,
        preprocessing_fn: Optional[callable] = None,
        dataset_kwargs: Optional[Dict[str, Any]] = None
    ) -> Dataset:
        """
        เตรียมชุดข้อมูลสำหรับการเทรนโมเดล
        
        Args:
            dataset (Union[Dataset, str]): ชุดข้อมูลหรือชื่อชุดข้อมูลใน Hugging Face Hub
            text_column (str): ชื่อคอลัมน์ที่มีข้อความสำหรับเทรน
            max_length (int): ความยาวสูงสุดของข้อความ
            preprocessing_fn (callable, optional): ฟังก์ชันสำหรับจัดการข้อมูลก่อนการเทรน
            dataset_kwargs (Dict[str, Any], optional): พารามิเตอร์เพิ่มเติมสำหรับการโหลดชุดข้อมูล
            
        Returns:
            Dataset: ชุดข้อมูลที่เตรียมพร้อมแล้ว
        """
        # โหลดชุดข้อมูลจาก Hugging Face หากเป็น string
        if isinstance(dataset, str):
            if dataset_kwargs is None:
                dataset_kwargs = {}
            dataset = load_dataset(dataset, **dataset_kwargs)
            
            # ถ้าโหลดข้อมูลมาหลายชุด ใช้ชุด train
            if isinstance(dataset, dict) and "train" in dataset:
                dataset = dataset["train"]
                
        # ตรวจสอบว่ามีคอลัมน์ข้อความหรือไม่
        if text_column not in dataset.column_names:
            available_columns = ", ".join(dataset.column_names)
            raise ValueError(f"ไม่พบคอลัมน์ '{text_column}' ในชุดข้อมูล คอลัมน์ที่มี: {available_columns}")
        
        # ถ้ามีฟังก์ชันจัดการข้อมูลเบื้องต้น ให้ใช้ก่อน
        if preprocessing_fn is not None:
            dataset = preprocessing_fn(dataset)
            
        # เตรียมข้อมูลด้วย tokenizer
        def tokenize_function(examples):
            return self.tokenizer(
                examples[text_column],
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors="pt"
            )
            
        # แปลงข้อมูลด้วย tokenizer
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=[col for col in dataset.column_names if col != text_column]
        )
            
        return tokenized_dataset
    
    def train(
        self,
        dataset: Dataset,
        output_dir: str = "./results",
        num_train_epochs: int = 3,
        per_device_train_batch_size: int = 8,
        gradient_accumulation_steps: int = 1,
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.1,
        save_steps: int = 500,
        eval_dataset: Optional[Dataset] = None,
        logging_steps: int = 50,
        evaluation_strategy: str = "steps",
        fp16: bool = True,
        push_to_hub: bool = False,
        hub_model_id: Optional[str] = None,
        hub_token: Optional[str] = None,
        **training_kwargs
    ):
        """
        เทรนโมเดลด้วยชุดข้อมูลที่เตรียมไว้
        
        Args:
            dataset (Dataset): ชุดข้อมูลที่เตรียมพร้อมแล้ว
            output_dir (str): ไดเรกทอรีสำหรับบันทึกผลลัพธ์
            num_train_epochs (int): จำนวนรอบการเทรน
            per_device_train_batch_size (int): ขนาดแบทช์ต่ออุปกรณ์
            gradient_accumulation_steps (int): จำนวนขั้นตอนสำหรับการสะสมเกรเดียนต์
            learning_rate (float): อัตราการเรียนรู้
            weight_decay (float): ค่าการลดน้ำหนัก
            warmup_ratio (float): สัดส่วนของขั้นตอนทั้งหมดที่ใช้สำหรับวอร์มอัป
            save_steps (int): บันทึกโมเดลทุกกี่ขั้นตอน
            eval_dataset (Dataset, optional): ชุดข้อมูลสำหรับการประเมินผล
            logging_steps (int): บันทึกข้อมูลทุกกี่ขั้นตอน
            evaluation_strategy (str): กลยุทธ์การประเมินผล ("no", "steps", "epoch")
            fp16 (bool): ใช้การคำนวณแบบ 16-bit floating point หรือไม่
            push_to_hub (bool): อัปโหลดโมเดลไปยัง Hugging Face Hub หรือไม่
            hub_model_id (str, optional): รหัสโมเดลบน Hub (ถ้าไม่ระบุจะใช้ชื่อ output_dir)
            hub_token (str, optional): โทเค็นสำหรับการอัปโหลดไปยัง Hub
            **training_kwargs: พารามิเตอร์เพิ่มเติมสำหรับการเทรน
            
        Returns:
            trainer: ออบเจ็กต์ Trainer ที่ใช้ในการเทรน
        """
        # สร้างไดเรกทอรีเก็บผลลัพธ์ถ้ายังไม่มี
        os.makedirs(output_dir, exist_ok=True)
        
        # เตรียม data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False  # เราไม่ได้ใช้ masked language modeling
        )
        
        # กำหนดพารามิเตอร์สำหรับการเทรน
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            warmup_ratio=warmup_ratio,
            logging_steps=logging_steps,
            save_steps=save_steps,
            evaluation_strategy=evaluation_strategy if eval_dataset is not None else "no",
            fp16=fp16 and self.device != "cpu",
            push_to_hub=push_to_hub,
            hub_model_id=hub_model_id,
            hub_token=hub_token,
            **training_kwargs
        )
        
        # สร้าง trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer
        )
        
        # เริ่มการเทรน
        trainer.train()
        
        # บันทึกโมเดลและ tokenizer
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        # อัปโหลดไปยัง Hub ถ้าต้องการ
        if push_to_hub:
            trainer.push_to_hub()
            
        return trainer
    
    def save_model(self, path: str):
        """
        บันทึกโมเดลและ tokenizer
        
        Args:
            path (str): พาธที่จะบันทึกโมเดล
        """
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        
    def load_model(self, path: str):
        """
        โหลดโมเดลและ tokenizer จากไฟล์ที่บันทึกไว้
        
        Args:
            path (str): พาธที่มีโมเดลที่บันทึกไว้
        """
        self.model = AutoModelForCausalLM.from_pretrained(
            path,
            device_map=self.device if self.device != "cpu" else None
        )
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        
    def generate(
        self,
        prompt: str,
        max_length: int = 100,
        num_return_sequences: int = 1,
        temperature: float = 1.0,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.0,
        do_sample: bool = True,
        **kwargs
    ) -> List[str]:
        """
        สร้างข้อความจากโมเดล
        
        Args:
            prompt (str): ข้อความเริ่มต้น
            max_length (int): ความยาวสูงสุดของข้อความที่สร้าง
            num_return_sequences (int): จำนวนข้อความที่จะสร้าง
            temperature (float): อุณหภูมิสำหรับการสุ่ม (ค่ายิ่งต่ำ ยิ่งแน่นอน)
            top_p (float): nucleus sampling (มากกว่า 0, น้อยกว่าหรือเท่ากับ 1)
            top_k (int): จำนวนโทเค็นที่มีความน่าจะเป็นสูงสุดที่จะพิจารณา
            repetition_penalty (float): บทลงโทษสำหรับการซ้ำคำ (มากกว่า 1.0 จะลงโทษการซ้ำ)
            do_sample (bool): สุ่มโทเค็นหรือไม่ (ถ้า False จะใช้ greedy decoding)
            **kwargs: พารามิเตอร์เพิ่มเติมสำหรับการสร้างข้อความ
            
        Returns:
            List[str]: รายการของข้อความที่สร้างขึ้น
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # ตั้งค่า generation
        generation_config = {
            "max_length": max_length,
            "num_return_sequences": num_return_sequences,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "repetition_penalty": repetition_penalty,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
            **kwargs
        }
        
        # สร้างข้อความ
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                **generation_config
            )
        
        # ถอดรหัสและคืนข้อความ
        generated_texts = [self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        
        return generated_texts 