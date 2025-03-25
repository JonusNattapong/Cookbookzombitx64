"""
โมดูลสำหรับสร้างชุดข้อมูลสำหรับการฝึกโมเดลเอไอ
"""

import os
import json
import random
import csv
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Union, Tuple, Optional, Callable
from datasets import Dataset, DatasetDict

class TextDatasetGenerator:
    """
    คลาสสำหรับสร้างชุดข้อมูลข้อความสำหรับการฝึกโมเดลภาษา
    """
    
    def __init__(self, seed: int = 42):
        """
        กำหนดค่าเริ่มต้นสำหรับตัวสร้างชุดข้อมูล
        
        Args:
            seed (int): ค่า seed สำหรับการสุ่ม เพื่อให้ผลลัพธ์คงที่
        """
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    def load_texts_from_files(self, directory: str, extension: str = ".txt") -> List[str]:
        """
        โหลดข้อความจากไฟล์ในไดเรกทอรี
        
        Args:
            directory (str): พาธของไดเรกทอรีที่มีไฟล์ข้อความ
            extension (str): นามสกุลของไฟล์ที่จะโหลด (ค่าเริ่มต้น: .txt)
            
        Returns:
            List[str]: รายการของข้อความที่โหลดได้
        """
        texts = []
        for file_path in Path(directory).glob(f"*{extension}"):
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                texts.append(content)
        return texts
    
    def load_texts_from_json(self, file_path: str, text_field: str = "text") -> List[str]:
        """
        โหลดข้อความจากไฟล์ JSON
        
        Args:
            file_path (str): พาธของไฟล์ JSON
            text_field (str): ชื่อฟิลด์ที่เก็บข้อความ (ค่าเริ่มต้น: "text")
            
        Returns:
            List[str]: รายการของข้อความที่โหลดได้
        """
        texts = []
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict) and text_field in item:
                        texts.append(item[text_field])
            elif isinstance(data, dict) and text_field in data:
                texts = data[text_field]
        return texts
    
    def load_texts_from_csv(self, file_path: str, text_column: str, delimiter: str = ',') -> List[str]:
        """
        โหลดข้อความจากไฟล์ CSV
        
        Args:
            file_path (str): พาธของไฟล์ CSV
            text_column (str): ชื่อคอลัมน์ที่เก็บข้อความ
            delimiter (str): ตัวคั่นข้อมูลในไฟล์ CSV (ค่าเริ่มต้น: ",")
            
        Returns:
            List[str]: รายการของข้อความที่โหลดได้
        """
        texts = []
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            for row in reader:
                if text_column in row:
                    texts.append(row[text_column])
        return texts
    
    def create_text_pairs(self, texts: List[str], min_length: int = 10) -> List[Tuple[str, str]]:
        """
        สร้างคู่ข้อความสำหรับการฝึกแบบ contrastive learning
        
        Args:
            texts (List[str]): รายการของข้อความ
            min_length (int): ความยาวขั้นต่ำของข้อความที่จะใช้
            
        Returns:
            List[Tuple[str, str]]: รายการของคู่ข้อความ (anchor, positive)
        """
        # กรองข้อความที่สั้นเกินไป
        filtered_texts = [text for text in texts if len(text) >= min_length]
        
        # สร้างคู่ข้อความ
        pairs = []
        for i, anchor in enumerate(filtered_texts):
            # เลือกข้อความที่เหลือเป็น positive
            for j, positive in enumerate(filtered_texts):
                if i != j:  # ไม่ใช้ข้อความเดียวกัน
                    pairs.append((anchor, positive))
                    
                    # จำกัดจำนวนคู่ต่อข้อความเพื่อไม่ให้มีมากเกินไป
                    if len(pairs) % 1000 == 0:
                        break
        
        # สลับคู่ข้อความ
        random.shuffle(pairs)
        
        return pairs
    
    def create_qa_pairs(self, questions: List[str], answers: List[str]) -> List[Dict[str, str]]:
        """
        สร้างคู่คำถาม-คำตอบสำหรับการฝึกแบบ QA
        
        Args:
            questions (List[str]): รายการของคำถาม
            answers (List[str]): รายการของคำตอบ
            
        Returns:
            List[Dict[str, str]]: รายการของคู่คำถาม-คำตอบ
        """
        if len(questions) != len(answers):
            raise ValueError("จำนวนคำถามและคำตอบต้องเท่ากัน")
            
        qa_pairs = []
        for q, a in zip(questions, answers):
            qa_pairs.append({"question": q, "answer": a})
            
        return qa_pairs
    
    def create_instruction_dataset(self, instructions: List[Dict[str, str]]) -> Dataset:
        """
        สร้างชุดข้อมูลแบบ instruction-based สำหรับการฝึกแบบ instruction tuning
        
        Args:
            instructions (List[Dict[str, str]]): รายการของคำสั่งพร้อมกับคำตอบ
                                               แต่ละรายการควรมีคีย์ 'instruction' และ 'response'
            
        Returns:
            Dataset: ชุดข้อมูลแบบ instruction-based
        """
        for instruction in instructions:
            if 'instruction' not in instruction or 'response' not in instruction:
                raise ValueError("แต่ละรายการต้องมีคีย์ 'instruction' และ 'response'")
                
        # แปลงเป็น Dataset
        dataset = Dataset.from_dict({
            'instruction': [item['instruction'] for item in instructions],
            'response': [item['response'] for item in instructions]
        })
        
        return dataset
    
    def create_preference_dataset(self, prompts: List[str], chosen: List[str], rejected: List[str]) -> Dataset:
        """
        สร้างชุดข้อมูลแบบ preference สำหรับการฝึกแบบ preference learning (RLHF)
        
        Args:
            prompts (List[str]): รายการของ prompt
            chosen (List[str]): รายการของคำตอบที่ถูกเลือก (preferred)
            rejected (List[str]): รายการของคำตอบที่ถูกปฏิเสธ (less preferred)
            
        Returns:
            Dataset: ชุดข้อมูลแบบ preference
        """
        if len(prompts) != len(chosen) or len(prompts) != len(rejected):
            raise ValueError("จำนวนของ prompts, chosen และ rejected ต้องเท่ากัน")
            
        # แปลงเป็น Dataset
        dataset = Dataset.from_dict({
            'prompt': prompts,
            'chosen': chosen,
            'rejected': rejected
        })
        
        return dataset
    
    def train_val_test_split(self, dataset: Dataset, train_size: float = 0.8, val_size: float = 0.1, 
                           test_size: float = 0.1, shuffle: bool = True) -> DatasetDict:
        """
        แบ่งชุดข้อมูลเป็นส่วนฝึก ส่วนตรวจสอบ และส่วนทดสอบ
        
        Args:
            dataset (Dataset): ชุดข้อมูลที่จะแบ่ง
            train_size (float): สัดส่วนของชุดข้อมูลฝึก (ค่าเริ่มต้น: 0.8)
            val_size (float): สัดส่วนของชุดข้อมูลตรวจสอบ (ค่าเริ่มต้น: 0.1)
            test_size (float): สัดส่วนของชุดข้อมูลทดสอบ (ค่าเริ่มต้น: 0.1)
            shuffle (bool): สุ่มข้อมูลก่อนแบ่งหรือไม่ (ค่าเริ่มต้น: True)
            
        Returns:
            DatasetDict: ชุดข้อมูลที่แบ่งแล้ว
        """
        if abs(train_size + val_size + test_size - 1.0) > 1e-6:
            raise ValueError("ผลรวมของ train_size, val_size และ test_size ต้องเท่ากับ 1")
            
        # สุ่มข้อมูล
        if shuffle:
            dataset = dataset.shuffle(seed=self.seed)
            
        # คำนวณจำนวนตัวอย่างในแต่ละส่วน
        total_size = len(dataset)
        train_samples = int(total_size * train_size)
        val_samples = int(total_size * val_size)
        
        # แบ่งชุดข้อมูล
        train_dataset = dataset.select(range(train_samples))
        val_dataset = dataset.select(range(train_samples, train_samples + val_samples))
        test_dataset = dataset.select(range(train_samples + val_samples, total_size))
        
        # สร้าง DatasetDict
        dataset_dict = DatasetDict({
            'train': train_dataset,
            'validation': val_dataset,
            'test': test_dataset
        })
        
        return dataset_dict
    
    def save_dataset(self, dataset: Union[Dataset, DatasetDict], output_dir: str, format: str = "arrow"):
        """
        บันทึกชุดข้อมูลลงในไฟล์
        
        Args:
            dataset (Union[Dataset, DatasetDict]): ชุดข้อมูลที่จะบันทึก
            output_dir (str): ไดเรกทอรีที่จะบันทึกชุดข้อมูล
            format (str): รูปแบบไฟล์ ("arrow", "json", "csv")
        """
        # สร้างไดเรกทอรีถ้ายังไม่มี
        os.makedirs(output_dir, exist_ok=True)
        
        # บันทึกชุดข้อมูล
        if isinstance(dataset, DatasetDict):
            for split, ds in dataset.items():
                if format == "json":
                    ds.to_json(os.path.join(output_dir, f"{split}.json"))
                elif format == "csv":
                    ds.to_csv(os.path.join(output_dir, f"{split}.csv"))
                else:  # ค่าเริ่มต้นเป็น arrow
                    ds.save_to_disk(os.path.join(output_dir, split))
        else:
            if format == "json":
                dataset.to_json(os.path.join(output_dir, "dataset.json"))
            elif format == "csv":
                dataset.to_csv(os.path.join(output_dir, "dataset.csv"))
            else:  # ค่าเริ่มต้นเป็น arrow
                dataset.save_to_disk(output_dir)

class ImageDatasetGenerator:
    """
    คลาสสำหรับสร้างชุดข้อมูลภาพสำหรับการฝึกโมเดลคอมพิวเตอร์วิชัน
    """
    
    def __init__(self, seed: int = 42):
        """
        กำหนดค่าเริ่มต้นสำหรับตัวสร้างชุดข้อมูลภาพ
        
        Args:
            seed (int): ค่า seed สำหรับการสุ่ม เพื่อให้ผลลัพธ์คงที่
        """
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    def load_images_from_directory(self, directory: str, extensions: List[str] = ['.jpg', '.jpeg', '.png']) -> Dict[str, str]:
        """
        โหลดภาพจากไดเรกทอรี
        
        Args:
            directory (str): พาธของไดเรกทอรีที่มีไฟล์ภาพ
            extensions (List[str]): รายการของนามสกุลไฟล์ที่จะโหลด
            
        Returns:
            Dict[str, str]: Dictionary ของชื่อไฟล์และพาธเต็ม
        """
        images = {}
        for ext in extensions:
            for file_path in Path(directory).glob(f"*{ext}"):
                images[file_path.name] = str(file_path)
        return images
    
    def load_images_with_labels(self, directory: str, extensions: List[str] = ['.jpg', '.jpeg', '.png']) -> Dict[str, List[Dict[str, str]]]:
        """
        โหลดภาพพร้อมป้ายกำกับจากโครงสร้างไดเรกทอรี
        
        Args:
            directory (str): พาธของไดเรกทอรีหลัก
            extensions (List[str]): รายการของนามสกุลไฟล์ที่จะโหลด
            
        Returns:
            Dict[str, List[Dict[str, str]]]: Dictionary ของป้ายกำกับและรายการของภาพ
        """
        labeled_images = {}
        
        # ในแต่ละโฟลเดอร์ย่อยเป็นป้ายกำกับหนึ่งประเภท
        for label_dir in Path(directory).iterdir():
            if label_dir.is_dir():
                label = label_dir.name
                labeled_images[label] = []
                
                # โหลดภาพในแต่ละโฟลเดอร์ย่อย
                for ext in extensions:
                    for file_path in label_dir.glob(f"*{ext}"):
                        labeled_images[label].append({
                            "file_name": file_path.name,
                            "file_path": str(file_path)
                        })
        
        return labeled_images
    
    def create_image_classification_dataset(self, labeled_images: Dict[str, List[Dict[str, str]]]) -> Dataset:
        """
        สร้างชุดข้อมูลสำหรับการจำแนกภาพ
        
        Args:
            labeled_images (Dict[str, List[Dict[str, str]]]): Dictionary ของป้ายกำกับและรายการของภาพ
            
        Returns:
            Dataset: ชุดข้อมูลสำหรับการจำแนกภาพ
        """
        image_paths = []
        labels = []
        
        # แปลงเป็นรายการของพาธภาพและป้ายกำกับ
        for label, images in labeled_images.items():
            for image in images:
                image_paths.append(image["file_path"])
                labels.append(label)
                
        # สร้าง Dataset
        dataset = Dataset.from_dict({
            "image_path": image_paths,
            "label": labels
        })
        
        return dataset
    
    def create_image_captioning_dataset(self, image_paths: List[str], captions: List[str]) -> Dataset:
        """
        สร้างชุดข้อมูลสำหรับการสร้างคำอธิบายภาพ
        
        Args:
            image_paths (List[str]): รายการของพาธภาพ
            captions (List[str]): รายการของคำอธิบายภาพ
            
        Returns:
            Dataset: ชุดข้อมูลสำหรับการสร้างคำอธิบายภาพ
        """
        if len(image_paths) != len(captions):
            raise ValueError("จำนวนของ image_paths และ captions ต้องเท่ากัน")
            
        # สร้าง Dataset
        dataset = Dataset.from_dict({
            "image_path": image_paths,
            "caption": captions
        })
        
        return dataset
    
    def train_val_test_split(self, dataset: Dataset, train_size: float = 0.8, val_size: float = 0.1, 
                           test_size: float = 0.1, shuffle: bool = True) -> DatasetDict:
        """
        แบ่งชุดข้อมูลเป็นส่วนฝึก ส่วนตรวจสอบ และส่วนทดสอบ
        
        Args:
            dataset (Dataset): ชุดข้อมูลที่จะแบ่ง
            train_size (float): สัดส่วนของชุดข้อมูลฝึก (ค่าเริ่มต้น: 0.8)
            val_size (float): สัดส่วนของชุดข้อมูลตรวจสอบ (ค่าเริ่มต้น: 0.1)
            test_size (float): สัดส่วนของชุดข้อมูลทดสอบ (ค่าเริ่มต้น: 0.1)
            shuffle (bool): สุ่มข้อมูลก่อนแบ่งหรือไม่ (ค่าเริ่มต้น: True)
            
        Returns:
            DatasetDict: ชุดข้อมูลที่แบ่งแล้ว
        """
        if abs(train_size + val_size + test_size - 1.0) > 1e-6:
            raise ValueError("ผลรวมของ train_size, val_size และ test_size ต้องเท่ากับ 1")
            
        # สุ่มข้อมูล
        if shuffle:
            dataset = dataset.shuffle(seed=self.seed)
            
        # คำนวณจำนวนตัวอย่างในแต่ละส่วน
        total_size = len(dataset)
        train_samples = int(total_size * train_size)
        val_samples = int(total_size * val_size)
        
        # แบ่งชุดข้อมูล
        train_dataset = dataset.select(range(train_samples))
        val_dataset = dataset.select(range(train_samples, train_samples + val_samples))
        test_dataset = dataset.select(range(train_samples + val_samples, total_size))
        
        # สร้าง DatasetDict
        dataset_dict = DatasetDict({
            'train': train_dataset,
            'validation': val_dataset,
            'test': test_dataset
        })
        
        return dataset_dict
    
    def save_dataset(self, dataset: Union[Dataset, DatasetDict], output_dir: str, format: str = "arrow"):
        """
        บันทึกชุดข้อมูลลงในไฟล์
        
        Args:
            dataset (Union[Dataset, DatasetDict]): ชุดข้อมูลที่จะบันทึก
            output_dir (str): ไดเรกทอรีที่จะบันทึกชุดข้อมูล
            format (str): รูปแบบไฟล์ ("arrow", "json", "csv")
        """
        # สร้างไดเรกทอรีถ้ายังไม่มี
        os.makedirs(output_dir, exist_ok=True)
        
        # บันทึกชุดข้อมูล
        if isinstance(dataset, DatasetDict):
            for split, ds in dataset.items():
                if format == "json":
                    ds.to_json(os.path.join(output_dir, f"{split}.json"))
                elif format == "csv":
                    ds.to_csv(os.path.join(output_dir, f"{split}.csv"))
                else:  # ค่าเริ่มต้นเป็น arrow
                    ds.save_to_disk(os.path.join(output_dir, split))
        else:
            if format == "json":
                dataset.to_json(os.path.join(output_dir, "dataset.json"))
            elif format == "csv":
                dataset.to_csv(os.path.join(output_dir, "dataset.csv"))
            else:  # ค่าเริ่มต้นเป็น arrow
                dataset.save_to_disk(output_dir) 