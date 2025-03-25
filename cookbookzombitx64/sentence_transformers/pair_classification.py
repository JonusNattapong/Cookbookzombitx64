"""
โมดูลสำหรับการปรับแต่งโมเดล Sentence Transformers สำหรับการจำแนกคู่ประโยค
"""

import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses, evaluation
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from typing import List, Dict, Optional, Union, Tuple, Any

class STPairClassificationTrainer:
    """คลาสสำหรับการปรับแต่งโมเดล Sentence Transformers สำหรับการจำแนกคู่ประโยค"""
    
    def __init__(
        self, 
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        num_labels: int = 2,
        device: Optional[str] = None
    ):
        """
        กำหนดค่าเริ่มต้นสำหรับการปรับแต่งโมเดล
        
        Args:
            model_name (str): ชื่อหรือพาธของโมเดล Sentence Transformer
            num_labels (int): จำนวนคลาสที่ต้องการจำแนก
            device (str, optional): อุปกรณ์ที่ใช้ในการคำนวณ ('cuda', 'cpu') ถ้าไม่ระบุจะเลือกอัตโนมัติ
        """
        # กำหนดอุปกรณ์ที่ใช้ในการคำนวณ
        if device:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
        # โหลดโมเดล
        self.model = SentenceTransformer(model_name, device=self.device)
        self.num_labels = num_labels
    
    def prepare_pairs(
        self, 
        pairs: List[Tuple[str, str]], 
        labels: List[int]
    ) -> List[InputExample]:
        """
        เตรียมคู่ประโยคสำหรับการเทรน
        
        Args:
            pairs (List[Tuple[str, str]]): รายการของคู่ประโยค (text1, text2)
            labels (List[int]): รายการของคลาสที่ต้องการจำแนก
            
        Returns:
            List[InputExample]: รายการของตัวอย่างข้อมูลสำหรับการเทรน
        """
        if len(pairs) != len(labels):
            raise ValueError("จำนวนของ pairs และ labels ต้องเท่ากัน")
            
        # ตรวจสอบค่า labels
        unique_labels = set(labels)
        if not all(0 <= label < self.num_labels for label in unique_labels):
            raise ValueError(f"ค่า labels ต้องอยู่ในช่วง 0 ถึง {self.num_labels-1}")
            
        # สร้างตัวอย่างข้อมูล
        examples = []
        for i, (pair, label) in enumerate(zip(pairs, labels)):
            examples.append(InputExample(texts=pair, label=label, guid=f"pair-{i}"))
            
        return examples
    
    def prepare_train_val_data(
        self, 
        train_examples: List[InputExample],
        val_examples: Optional[List[InputExample]] = None,
        val_split: float = 0.1,
        batch_size: int = 16,
        shuffle: bool = True
    ) -> Tuple[DataLoader, Optional[evaluation.BinaryClassificationEvaluator]]:
        """
        เตรียม DataLoader สำหรับการเทรนและข้อมูลสำหรับการประเมินผล
        
        Args:
            train_examples (List[InputExample]): รายการของตัวอย่างข้อมูลสำหรับการเทรน
            val_examples (List[InputExample], optional): รายการของตัวอย่างข้อมูลสำหรับการตรวจสอบ
            val_split (float): สัดส่วนของข้อมูลที่จะใช้เป็นชุดตรวจสอบ (ถ้าไม่มี val_examples)
            batch_size (int): ขนาดของแบตช์
            shuffle (bool): สลับข้อมูลหรือไม่
            
        Returns:
            Tuple[DataLoader, Optional[evaluation.BinaryClassificationEvaluator]]: DataLoader และตัวประเมินผล
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
            val_labels = [example.label for example in val_examples]
            
            val_evaluator = evaluation.BinaryClassificationEvaluator(
                val_sentences1, val_sentences2, val_labels
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
        if self.num_labels == 2:
            train_loss = losses.SoftmaxLoss(
                model=self.model,
                sentence_embedding_dimension=self.model.get_sentence_embedding_dimension(),
                num_labels=self.num_labels
            )
        else:
            train_loss = losses.MultipleNegativesRankingLoss(self.model)
        
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
        
    def predict(
        self, 
        pairs: List[Tuple[str, str]], 
        batch_size: int = 32
    ) -> Tuple[List[int], List[List[float]]]:
        """
        ทำนายคลาสของคู่ประโยค
        
        Args:
            pairs (List[Tuple[str, str]]): รายการของคู่ประโยค (text1, text2)
            batch_size (int): ขนาดของแบตช์
            
        Returns:
            Tuple[List[int], List[List[float]]]: คลาสที่ทำนายและความน่าจะเป็นของแต่ละคลาส
        """
        # แยกคู่ประโยค
        sentences1 = [pair[0] for pair in pairs]
        sentences2 = [pair[1] for pair in pairs]
        
        # แปลงประโยคเป็น embeddings
        embeddings1 = self.model.encode(sentences1, batch_size=batch_size, convert_to_tensor=True)
        embeddings2 = self.model.encode(sentences2, batch_size=batch_size, convert_to_tensor=True)
        
        # คำนวณความคล้ายคลึง
        cosine_scores = F.cosine_similarity(embeddings1, embeddings2)
        
        # แปลงเป็นความน่าจะเป็น
        probabilities = F.softmax(cosine_scores.unsqueeze(1), dim=1)
        
        # ทำนายคลาส
        predictions = torch.argmax(probabilities, dim=1)
        
        return predictions.tolist(), probabilities.tolist()
    
    def evaluate(
        self, 
        pairs: List[Tuple[str, str]], 
        true_labels: List[int]
    ) -> Dict[str, float]:
        """
        ประเมินผลโมเดล
        
        Args:
            pairs (List[Tuple[str, str]]): รายการของคู่ประโยค (text1, text2)
            true_labels (List[int]): รายการของคลาสที่ถูกต้อง
            
        Returns:
            Dict[str, float]: ผลการประเมิน (accuracy, precision, recall, f1-score)
        """
        # ทำนายคลาส
        predictions, _ = self.predict(pairs)
        
        # คำนวณ metrics
        accuracy = accuracy_score(true_labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, predictions, average='weighted'
        )
        
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        } 