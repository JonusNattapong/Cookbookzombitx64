"""
โมดูลสำหรับการปรับแต่งโมเดล Sentence Transformers สำหรับการให้คะแนนคู่ประโยค
"""

import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses, evaluation
from sklearn.metrics import mean_squared_error, mean_absolute_error
from typing import List, Dict, Optional, Union, Tuple, Any

class STPairScoringTrainer:
    """คลาสสำหรับการปรับแต่งโมเดล Sentence Transformers สำหรับการให้คะแนนคู่ประโยค"""
    
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
        scores: List[float]
    ) -> List[InputExample]:
        """
        เตรียมคู่ประโยคสำหรับการเทรน
        
        Args:
            pairs (List[Tuple[str, str]]): รายการของคู่ประโยค (text1, text2)
            scores (List[float]): รายการของคะแนนความคล้ายคลึง
            
        Returns:
            List[InputExample]: รายการของตัวอย่างข้อมูลสำหรับการเทรน
        """
        if len(pairs) != len(scores):
            raise ValueError("จำนวนของ pairs และ scores ต้องเท่ากัน")
            
        # สร้างตัวอย่างข้อมูล
        examples = []
        for i, (pair, score) in enumerate(zip(pairs, scores)):
            examples.append(InputExample(texts=pair, label=float(score), guid=f"pair-{i}"))
            
        return examples
    
    def prepare_train_val_data(
        self, 
        train_examples: List[InputExample],
        val_examples: Optional[List[InputExample]] = None,
        val_split: float = 0.1,
        batch_size: int = 16,
        shuffle: bool = True
    ) -> Tuple[DataLoader, Optional[evaluation.EmbeddingSimilarityEvaluator]]:
        """
        เตรียม DataLoader สำหรับการเทรนและข้อมูลสำหรับการประเมินผล
        
        Args:
            train_examples (List[InputExample]): รายการของตัวอย่างข้อมูลสำหรับการเทรน
            val_examples (List[InputExample], optional): รายการของตัวอย่างข้อมูลสำหรับการตรวจสอบ
            val_split (float): สัดส่วนของข้อมูลที่จะใช้เป็นชุดตรวจสอบ (ถ้าไม่มี val_examples)
            batch_size (int): ขนาดของแบตช์
            shuffle (bool): สลับข้อมูลหรือไม่
            
        Returns:
            Tuple[DataLoader, Optional[evaluation.EmbeddingSimilarityEvaluator]]: DataLoader และตัวประเมินผล
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
        
    def predict_scores(
        self, 
        pairs: List[Tuple[str, str]], 
        batch_size: int = 32,
        normalize: bool = True
    ) -> List[float]:
        """
        ทำนายคะแนนความคล้ายคลึงของคู่ประโยค
        
        Args:
            pairs (List[Tuple[str, str]]): รายการของคู่ประโยค (text1, text2)
            batch_size (int): ขนาดของแบตช์
            normalize (bool): ปรับค่าให้อยู่ในช่วง 0-1 หรือไม่
            
        Returns:
            List[float]: รายการของคะแนนความคล้ายคลึง
        """
        # แยกคู่ประโยค
        sentences1 = [pair[0] for pair in pairs]
        sentences2 = [pair[1] for pair in pairs]
        
        # แปลงประโยคเป็น embeddings
        embeddings1 = self.model.encode(sentences1, batch_size=batch_size, convert_to_tensor=True)
        embeddings2 = self.model.encode(sentences2, batch_size=batch_size, convert_to_tensor=True)
        
        # คำนวณความคล้ายคลึง
        cosine_scores = F.cosine_similarity(embeddings1, embeddings2)
        
        # ปรับค่าให้อยู่ในช่วง 0-1 ถ้าต้องการ
        if normalize:
            scores = (cosine_scores + 1) / 2
        else:
            scores = cosine_scores
            
        return scores.tolist()
    
    def evaluate(
        self, 
        pairs: List[Tuple[str, str]], 
        true_scores: List[float],
        normalize: bool = True
    ) -> Dict[str, float]:
        """
        ประเมินผลโมเดล
        
        Args:
            pairs (List[Tuple[str, str]]): รายการของคู่ประโยค (text1, text2)
            true_scores (List[float]): รายการของคะแนนที่ถูกต้อง
            normalize (bool): ปรับค่าให้อยู่ในช่วง 0-1 หรือไม่
            
        Returns:
            Dict[str, float]: ผลการประเมิน (MSE, MAE, Pearson correlation)
        """
        # ทำนายคะแนน
        predicted_scores = self.predict_scores(pairs, normalize=normalize)
        
        # คำนวณ metrics
        mse = mean_squared_error(true_scores, predicted_scores)
        mae = mean_absolute_error(true_scores, predicted_scores)
        
        # คำนวณ Pearson correlation
        correlation = torch.corrcoef(
            torch.tensor([true_scores, predicted_scores])
        )[0, 1].item()
        
        return {
            "mse": mse,
            "mae": mae,
            "pearson_correlation": correlation
        } 