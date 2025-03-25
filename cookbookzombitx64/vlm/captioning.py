"""
โมดูลสำหรับการปรับแต่งโมเดล Vision Language Model (VLM) สำหรับการสร้างคำบรรยายภาพ
"""

import os
import torch
from transformers import (
    VisionEncoderDecoderModel, 
    AutoTokenizer, 
    ViTImageProcessor, 
    Trainer, 
    TrainingArguments
)
from datasets import Dataset
from PIL import Image
from typing import List, Dict, Optional, Union, Tuple

class VLMCaptioningTrainer:
    """คลาสสำหรับการปรับแต่งโมเดล VLM สำหรับการสร้างคำบรรยายภาพ"""
    
    def __init__(
        self, 
        model_name: str = "nlpconnect/vit-gpt2-image-captioning", 
        encoder_name: Optional[str] = None,
        decoder_name: Optional[str] = None,
        tokenizer_name: Optional[str] = None,
        device: Optional[str] = None
    ):
        """
        กำหนดค่าเริ่มต้นสำหรับการปรับแต่งโมเดล VLM
        
        Args:
            model_name (str): ชื่อหรือพาธของโมเดล VLM สำเร็จรูป 
                             (หรือจะระบุ encoder_name และ decoder_name แยกกัน)
            encoder_name (str, optional): ชื่อหรือพาธของ visual encoder ถ้าต้องการสร้างโมเดลใหม่
            decoder_name (str, optional): ชื่อหรือพาธของ text decoder ถ้าต้องการสร้างโมเดลใหม่
            tokenizer_name (str, optional): ชื่อหรือพาธของ tokenizer ถ้าไม่ระบุจะใช้จาก decoder_name
            device (str, optional): อุปกรณ์ที่ใช้ในการคำนวณ ('cuda', 'cpu') ถ้าไม่ระบุจะเลือกอัตโนมัติ
        """
        # กำหนดอุปกรณ์ที่ใช้ในการคำนวณ
        if device:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
        if encoder_name and decoder_name:
            # สร้างโมเดลใหม่จาก encoder และ decoder ที่กำหนด
            self.model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
                encoder_name, decoder_name
            ).to(self.device)
            
            # กำหนด tokenizer จาก decoder
            tokenizer_name = tokenizer_name or decoder_name
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            
            # กำหนด image processor จาก encoder
            self.image_processor = ViTImageProcessor.from_pretrained(encoder_name)
        else:
            # โหลดโมเดลสำเร็จรูป
            self.model = VisionEncoderDecoderModel.from_pretrained(model_name).to(self.device)
            
            # ลองโหลด tokenizer และ image processor ที่เหมาะสม
            tokenizer_name = tokenizer_name or model_name
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            
            # ลองโหลด image processor
            try:
                self.image_processor = ViTImageProcessor.from_pretrained(model_name)
            except:
                # ถ้าไม่สามารถโหลดได้ ลองใช้ ViT default
                self.image_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
        
        # ตั้งค่า special tokens
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # ตั้งค่า generation parameters
        self.model.config.decoder_start_token_id = self.tokenizer.cls_token_id if hasattr(self.tokenizer, 'cls_token_id') else self.tokenizer.bos_token_id
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.config.vocab_size = self.tokenizer.vocab_size
    
    def prepare_dataset(self, image_paths: List[str], captions: List[str], max_length: int = 64) -> Dataset:
        """
        เตรียมชุดข้อมูลสำหรับการเทรน
        
        Args:
            image_paths (List[str]): รายการของพาธไฟล์ภาพ
            captions (List[str]): รายการของคำบรรยายภาพ
            max_length (int): ความยาวสูงสุดของคำบรรยาย
            
        Returns:
            Dataset: ชุดข้อมูลที่เตรียมพร้อมแล้ว
        """
        if len(image_paths) != len(captions):
            raise ValueError("จำนวนของ image_paths และ captions ต้องเท่ากัน")
            
        # สร้างชุดข้อมูล
        data = {
            "image_path": image_paths,
            "caption": captions
        }
        
        dataset = Dataset.from_dict(data)
        
        # เตรียมข้อมูลสำหรับการเทรน
        def preprocess_data(examples):
            # โหลดภาพ
            images = [Image.open(image_path).convert("RGB") for image_path in examples["image_path"]]
            
            # แปลงภาพด้วย image processor
            pixel_values = self.image_processor(images, return_tensors="pt").pixel_values
            
            # แปลงคำบรรยายด้วย tokenizer
            tokenized_captions = self.tokenizer(
                examples["caption"], 
                padding="max_length", 
                truncation=True, 
                max_length=max_length, 
                return_tensors="pt"
            )
            
            # เตรียมข้อมูลสำหรับโมเดล
            return {
                "pixel_values": pixel_values,
                "labels": tokenized_captions["input_ids"],
                "decoder_attention_mask": tokenized_captions["attention_mask"]
            }
        
        processed_dataset = dataset.map(preprocess_data, batched=True)
        processed_dataset.set_format(type="torch", columns=["pixel_values", "labels", "decoder_attention_mask"])
        
        return processed_dataset
    
    def train(
        self, 
        dataset: Dataset, 
        output_dir: str = "./results", 
        num_train_epochs: int = 3, 
        per_device_train_batch_size: int = 4, 
        learning_rate: float = 5e-5, 
        weight_decay: float = 0.01, 
        save_steps: int = 1000, 
        eval_dataset: Optional[Dataset] = None
    ):
        """
        เทรนโมเดล
        
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
        # กำหนดค่าพารามิเตอร์สำหรับการเทรน
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            save_steps=save_steps,
            logging_dir=f"{output_dir}/logs",
            remove_unused_columns=False,
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
        บันทึกโมเดลและ tokenizer
        
        Args:
            path (str): พาธที่จะบันทึกโมเดล
        """
        # สร้างไดเรกทอรีถ้ายังไม่มี
        os.makedirs(path, exist_ok=True)
        
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        self.image_processor.save_pretrained(path)
        
    def generate_caption(self, image_path: str, max_length: int = 64, num_beams: int = 4) -> str:
        """
        สร้างคำบรรยายสำหรับภาพ
        
        Args:
            image_path (str): พาธของไฟล์ภาพ
            max_length (int): ความยาวสูงสุดของคำบรรยาย
            num_beams (int): จำนวน beams สำหรับการค้นหาแบบ beam search
            
        Returns:
            str: คำบรรยายที่สร้างขึ้น
        """
        # โหลดภาพ
        image = Image.open(image_path).convert("RGB")
        
        # แปลงภาพด้วย image processor
        pixel_values = self.image_processor(image, return_tensors="pt").pixel_values.to(self.device)
        
        # สร้างคำบรรยาย
        output_ids = self.model.generate(
            pixel_values,
            max_length=max_length,
            num_beams=num_beams,
            early_stopping=True
        )
        
        # แปลงคำบรรยายกลับเป็นข้อความ
        caption = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        return caption