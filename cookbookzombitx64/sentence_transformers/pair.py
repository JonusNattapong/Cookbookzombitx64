"""
โมดูลสำหรับการปรับแต่งโมเดล Sentence Transformers ด้วยวิธีการฝึกแบบ Pair
"""

import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses, evaluation
from typing import List, Dict, Optional, Union, Tuple, Any

class STPairTrainer:
    """คลาสสำหรับการปรับแต่งโมเดล Sentence Transformers ด้วยวิธีการฝึกแบบ Pair"""
    
    def __init__(
        self, 
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: Optional[str] = None
    ):
        """
        กำหนดค่าเริ่มต้นสำหรับการปรับแต่งโมเดล
        
        Args:
            model_name (str): ชื่อหรือพาธของโมเดล Sentence Transformer
            device (str, optional): อุปกรณ์ที่ใช้ในการคำนวณ ('cuda', 'cpu') ถ้าไม่ระบุจะเลือกอัตโนมัติ
        """
        # กำหนดอุปกรณ์ที่ใช้ในการคำนวณ
        if device:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
        # โหลดโมเดล
        self.model = SentenceTransformer(model_name, device=self.device)
    
    def prepare_pairs(
        self, 
        pairs: List[Tuple[str, str]], 
        similarity_scores: Optional[List[float]] = None
    ) -> List[InputExample]:
        """
        เตรียมคู่ประโยคสำหรับการเทรน
        
        Args:
            pairs (List[Tuple[str, str]]): รายการของคู่ประโยค (text1, text2)
            similarity_scores (List[float], optional): คะแนนความคล้ายคลึงของแต่ละคู่ (0-1)
                                                     ถ้าไม่ระบุจะกำหนดค่าเป็น 1.0 (คู่ที่เหมือนกัน)
            
        Returns:
            List[InputExample]: รายการของตัวอย่างข้อมูลสำหรับการเทรน
        """
        examples = []
        
        # กำหนดค่า similarity_scores ถ้าไม่ได้ระบุ
        if similarity_scores is None:
            similarity_scores = [1.0] * len(pairs)
        elif len(pairs) != len(similarity_scores):
            raise ValueError("จำนวนของ pairs และ similarity_scores ต้องเท่ากัน")
            
        # สร้างตัวอย่างข้อมูล
        for i, (pair, score) in enumerate(zip(pairs, similarity_scores)):
            examples.append(InputExample(texts=pair, label=score, guid=f"pair-{i}"))
            
        return examples
    
    def prepare_train_val_data(
        self, 
        train_examples: List[InputExample],
        val_examples: Optional[List[InputExample]] = None,
        val_split: float = 0.1,
        batch_size: int = 16,
        shuffle: bool = True
    ) -> Tuple[DataLoader, Optional[List[Tuple[str, str]]]]:
        """
        เตรียม DataLoader สำหรับการเทรนและข้อมูลสำหรับการประเมินผล
        
        Args:
            train_examples (List[InputExample]): รายการของตัวอย่างข้อมูลสำหรับการเทรน
            val_examples (List[InputExample], optional): รายการของตัวอย่างข้อมูลสำหรับการตรวจสอบ
            val_split (float): สัดส่วนของข้อมูลที่จะใช้เป็นชุดตรวจสอบ (ถ้าไม่มี val_examples)
            batch_size (int): ขนาดของแบตช์
            shuffle (bool): สลับข้อมูลหรือไม่
            
        Returns:
            Tuple[DataLoader, Optional[List[Tuple[str, str]]]]: DataLoader สำหรับการเทรนและข้อมูลสำหรับการประเมินผล
        """
        # ถ้าไม่มีข้อมูลตรวจสอบและต้องการแบ่งจากข้อมูลเทรน
        if val_examples is None and val_split > 0:
            val_size = int(len(train_examples) * val_split)
            val_examples = train_examples[-val_size:]
            train_examples = train_examples[:-val_size]
            
        # สร้าง DataLoader
        train_dataloader = DataLoader(train_examples, batch_size=batch_size, shuffle=shuffle)
        
        # เตรียมข้อมูลสำหรับการประเมินผล
        val_evaluator = None
        if val_examples:
            val_sentences1 = [example.texts[0] for example in val_examples]
            val_sentences2 = [example.texts[1] for example in val_examples]
            val_scores = [example.label for example in val_examples]
            
            val_evaluator = evaluation.EmbeddingSimilarityEvaluator(
                val_sentences1, val_sentences2, val_scores
            )
            
        return train_dataloader, val_evaluator
    
    def train(
        self, 
        train_dataloader: DataLoader,
        val_evaluator: Optional[Any] = None,
        epochs: int = 10,
        warmup_steps: int = 100,
        evaluation_steps: int = 1000,
        output_dir: str = "./results"
    ):
        """
        เทรนโมเดล
        
        Args:
            train_dataloader (DataLoader): DataLoader สำหรับการเทรน
            val_evaluator (Any, optional): ตัวประเมินผลสำหรับการตรวจสอบ
            epochs (int): จำนวนรอบการเทรน
            warmup_steps (int): จำนวนขั้นตอนของ warmup
            evaluation_steps (int): จำนวนขั้นตอนก่อนที่จะประเมินผล
            output_dir (str): ไดเรกทอรีสำหรับบันทึกผลลัพธ์
        """
        # กำหนด loss function
        train_loss = losses.CosineSimilarityLoss(self.model)
        
        # สร้างไดเรกทอรีถ้ายังไม่มี
        os.makedirs(output_dir, exist_ok=True)
        
        # เริ่มการเทรน
        self.model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            evaluator=val_evaluator,
            epochs=epochs,
            warmup_steps=warmup_steps,
            evaluation_steps=evaluation_steps,
            output_path=output_dir,
            save_best_model=True
        )
        
        return self.model
    
    def save_model(self, path: str):
        """
        บันทึกโมเดล
        
        Args:
            path (str): พาธที่จะบันทึกโมเดล
        """
        self.model.save(path)
        
    def encode(self, sentences: Union[str, List[str]], batch_size: int = 32, **kwargs) -> torch.Tensor:
        """
        แปลงประโยคเป็น embeddings
        
        Args:
            sentences (Union[str, List[str]]): ประโยคหรือรายการของประโยค
            batch_size (int): ขนาดของแบตช์
            **kwargs: พารามิเตอร์เพิ่มเติมสำหรับการแปลง
            
        Returns:
            torch.Tensor: embedding ของประโยค
        """
        return self.model.encode(sentences, batch_size=batch_size, **kwargs)
    
    def calculate_similarity(self, sentence1: str, sentence2: str) -> float:
        """
        คำนวณความคล้ายคลึงระหว่างสองประโยค
        
        Args:
            sentence1 (str): ประโยคที่หนึ่ง
            sentence2 (str): ประโยคที่สอง
            
        Returns:
            float: คะแนนความคล้ายคลึง (ค่าระหว่าง -1 ถึง 1)
        """
        # แปลงประโยคเป็น embeddings
        embedding1 = self.encode(sentence1)
        embedding2 = self.encode(sentence2)
        
        # คำนวณความคล้ายคลึงด้วย cosine similarity
        similarity = F.cosine_similarity(
            torch.tensor(embedding1).unsqueeze(0),
            torch.tensor(embedding2).unsqueeze(0)
        ).item()
        
        return similarity 