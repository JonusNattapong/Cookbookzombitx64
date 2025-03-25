"""
โมดูลสำหรับการปรับแต่งโมเดล Sentence Transformers ด้วยวิธีการฝึกแบบ Triplet
"""

import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses, evaluation
from typing import List, Dict, Optional, Union, Tuple, Any

class STTripletTrainer:
    """คลาสสำหรับการปรับแต่งโมเดล Sentence Transformers ด้วยวิธีการฝึกแบบ Triplet"""
    
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
    
    def prepare_triplets(
        self, 
        triplets: List[Tuple[str, str, str]], 
        margin: float = 1.0
    ) -> List[InputExample]:
        """
        เตรียม triplets สำหรับการเทรน
        
        Args:
            triplets (List[Tuple[str, str, str]]): รายการของ triplets (anchor, positive, negative)
            margin (float): ค่า margin สำหรับ triplet loss
            
        Returns:
            List[InputExample]: รายการของตัวอย่างข้อมูลสำหรับการเทรน
        """
        examples = []
        
        # สร้างตัวอย่างข้อมูล
        for i, (anchor, positive, negative) in enumerate(triplets):
            examples.append(InputExample(
                texts=[anchor, positive, negative],
                label=margin,  # ใช้ margin เป็น label
                guid=f"triplet-{i}"
            ))
            
        return examples
    
    def prepare_train_val_data(
        self, 
        train_examples: List[InputExample],
        val_examples: Optional[List[InputExample]] = None,
        val_split: float = 0.1,
        batch_size: int = 16,
        shuffle: bool = True
    ) -> Tuple[DataLoader, Optional[evaluation.TripletEvaluator]]:
        """
        เตรียม DataLoader สำหรับการเทรนและข้อมูลสำหรับการประเมินผล
        
        Args:
            train_examples (List[InputExample]): รายการของตัวอย่างข้อมูลสำหรับการเทรน
            val_examples (List[InputExample], optional): รายการของตัวอย่างข้อมูลสำหรับการตรวจสอบ
            val_split (float): สัดส่วนของข้อมูลที่จะใช้เป็นชุดตรวจสอบ (ถ้าไม่มี val_examples)
            batch_size (int): ขนาดของแบตช์
            shuffle (bool): สลับข้อมูลหรือไม่
            
        Returns:
            Tuple[DataLoader, Optional[evaluation.TripletEvaluator]]: DataLoader และตัวประเมินผล
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
            val_anchors = [example.texts[0] for example in val_examples]
            val_positives = [example.texts[1] for example in val_examples]
            val_negatives = [example.texts[2] for example in val_examples]
            
            val_evaluator = evaluation.TripletEvaluator(
                anchors=val_anchors,
                positives=val_positives,
                negatives=val_negatives
            )
            
        return train_dataloader, val_evaluator
    
    def train(
        self, 
        train_dataloader: DataLoader,
        val_evaluator: Optional[Any] = None,
        epochs: int = 10,
        warmup_steps: int = 100,
        evaluation_steps: int = 1000,
        margin: float = 1.0,
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
            margin (float): ค่า margin สำหรับ triplet loss
            output_dir (str): ไดเรกทอรีสำหรับบันทึกผลลัพธ์
        """
        # กำหนด loss function
        train_loss = losses.TripletLoss(
            model=self.model,
            distance_metric=losses.TripletDistanceMetric.COSINE,
            triplet_margin=margin
        )
        
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
    
    def evaluate_triplet(
        self, 
        triplets: List[Tuple[str, str, str]], 
        batch_size: int = 32
    ) -> Dict[str, float]:
        """
        ประเมินผลโมเดลด้วย triplets
        
        Args:
            triplets (List[Tuple[str, str, str]]): รายการของ triplets (anchor, positive, negative)
            batch_size (int): ขนาดของแบตช์
            
        Returns:
            Dict[str, float]: ผลการประเมิน (accuracy, average_margin)
        """
        # แยก triplets
        anchors = [t[0] for t in triplets]
        positives = [t[1] for t in triplets]
        negatives = [t[2] for t in triplets]
        
        # แปลงประโยคเป็น embeddings
        anchor_embeddings = self.encode(anchors, batch_size=batch_size, convert_to_tensor=True)
        positive_embeddings = self.encode(positives, batch_size=batch_size, convert_to_tensor=True)
        negative_embeddings = self.encode(negatives, batch_size=batch_size, convert_to_tensor=True)
        
        # คำนวณความคล้ายคลึง
        pos_scores = F.cosine_similarity(anchor_embeddings, positive_embeddings)
        neg_scores = F.cosine_similarity(anchor_embeddings, negative_embeddings)
        
        # คำนวณ margin
        margins = pos_scores - neg_scores
        
        # คำนวณ accuracy
        accuracy = (margins > 0).float().mean().item()
        
        # คำนวณค่าเฉลี่ยของ margin
        average_margin = margins.mean().item()
        
        return {
            "accuracy": accuracy,
            "average_margin": average_margin
        }
    
    def find_nearest_neighbors(
        self, 
        query: str, 
        corpus: List[str], 
        k: int = 5, 
        batch_size: int = 32
    ) -> List[Dict[str, Any]]:
        """
        ค้นหาประโยคที่ใกล้เคียงที่สุด k ประโยคจากคลังข้อมูล
        
        Args:
            query (str): ประโยคที่ต้องการค้นหา
            corpus (List[str]): คลังข้อมูลที่ต้องการค้นหา
            k (int): จำนวนประโยคที่ต้องการค้นหา
            batch_size (int): ขนาดของแบตช์
            
        Returns:
            List[Dict[str, Any]]: รายการของประโยคที่ใกล้เคียงที่สุดพร้อมคะแนน
        """
        # แปลงประโยคเป็น embeddings
        query_embedding = self.encode(query, convert_to_tensor=True)
        corpus_embeddings = self.encode(corpus, batch_size=batch_size, convert_to_tensor=True)
        
        # คำนวณความคล้ายคลึง
        cos_scores = F.cosine_similarity(query_embedding, corpus_embeddings)
        
        # เรียงลำดับและเลือก k อันดับแรก
        top_results = torch.topk(cos_scores, k=min(k, len(corpus)))
        
        results = []
        for score, idx in zip(top_results.values, top_results.indices):
            results.append({
                "text": corpus[idx],
                "score": score.item()
            })
            
        return results 