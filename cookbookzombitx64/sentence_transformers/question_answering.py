"""
โมดูลสำหรับการปรับแต่งโมเดล Sentence Transformers สำหรับการตอบคำถาม
"""

import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses, evaluation
from typing import List, Dict, Optional, Union, Tuple, Any

class STQuestionAnsweringTrainer:
    """คลาสสำหรับการปรับแต่งโมเดล Sentence Transformers สำหรับการตอบคำถาม"""
    
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
    
    def prepare_qa_pairs(
        self, 
        questions: List[str], 
        answers: List[str], 
        negative_answers: Optional[List[List[str]]] = None,
        num_negatives: int = 5
    ) -> List[InputExample]:
        """
        เตรียมคู่คำถาม-คำตอบสำหรับการเทรน
        
        Args:
            questions (List[str]): รายการของคำถาม
            answers (List[str]): รายการของคำตอบที่ถูกต้อง
            negative_answers (List[List[str]], optional): รายการของคำตอบที่ไม่ถูกต้องสำหรับแต่ละคำถาม
            num_negatives (int): จำนวนคำตอบที่ไม่ถูกต้องที่ต้องการสร้างถ้าไม่ได้ระบุ negative_answers
            
        Returns:
            List[InputExample]: รายการของตัวอย่างข้อมูลสำหรับการเทรน
        """
        if len(questions) != len(answers):
            raise ValueError("จำนวนของ questions และ answers ต้องเท่ากัน")
            
        examples = []
        
        # ถ้าไม่มี negative_answers ให้ใช้คำตอบอื่นๆ เป็น negative
        if negative_answers is None:
            for i, (question, answer) in enumerate(zip(questions, answers)):
                # สร้าง negative answers โดยสุ่มจากคำตอบอื่นๆ
                other_answers = [a for j, a in enumerate(answers) if j != i]
                if len(other_answers) >= num_negatives:
                    neg_answers = other_answers[:num_negatives]
                else:
                    # ถ้ามีคำตอบไม่พอ ให้ใช้คำตอบซ้ำ
                    neg_answers = other_answers * (num_negatives // len(other_answers) + 1)
                    neg_answers = neg_answers[:num_negatives]
                    
                # สร้างตัวอย่างข้อมูล
                examples.append(InputExample(
                    texts=[question, answer] + neg_answers,
                    label=1.0,  # คะแนนสำหรับคำตอบที่ถูกต้อง
                    guid=f"qa-{i}"
                ))
        else:
            # ใช้ negative_answers ที่กำหนดมา
            if len(questions) != len(negative_answers):
                raise ValueError("จำนวนของ questions และ negative_answers ต้องเท่ากัน")
                
            for i, (question, answer, negs) in enumerate(zip(questions, answers, negative_answers)):
                examples.append(InputExample(
                    texts=[question, answer] + negs,
                    label=1.0,
                    guid=f"qa-{i}"
                ))
                
        return examples
    
    def prepare_train_val_data(
        self, 
        train_examples: List[InputExample],
        val_examples: Optional[List[InputExample]] = None,
        val_split: float = 0.1,
        batch_size: int = 16,
        shuffle: bool = True
    ) -> Tuple[DataLoader, Optional[evaluation.RerankingEvaluator]]:
        """
        เตรียม DataLoader สำหรับการเทรนและข้อมูลสำหรับการประเมินผล
        
        Args:
            train_examples (List[InputExample]): รายการของตัวอย่างข้อมูลสำหรับการเทรน
            val_examples (List[InputExample], optional): รายการของตัวอย่างข้อมูลสำหรับการตรวจสอบ
            val_split (float): สัดส่วนของข้อมูลที่จะใช้เป็นชุดตรวจสอบ (ถ้าไม่มี val_examples)
            batch_size (int): ขนาดของแบตช์
            shuffle (bool): สลับข้อมูลหรือไม่
            
        Returns:
            Tuple[DataLoader, Optional[evaluation.RerankingEvaluator]]: DataLoader และตัวประเมินผล
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
            val_queries = [example.texts[0] for example in val_examples]
            val_relevant_docs = [[example.texts[1]] for example in val_examples]
            val_irrelevant_docs = [example.texts[2:] for example in val_examples]
            
            val_evaluator = evaluation.RerankingEvaluator(
                queries=val_queries,
                relevant_docs=val_relevant_docs,
                irrelevant_docs=val_irrelevant_docs
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
        
    def find_answers(
        self, 
        question: str, 
        candidate_answers: List[str], 
        top_k: int = 5, 
        threshold: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        ค้นหาคำตอบที่ดีที่สุดจากคำตอบที่เป็นไปได้
        
        Args:
            question (str): คำถาม
            candidate_answers (List[str]): รายการของคำตอบที่เป็นไปได้
            top_k (int): จำนวนคำตอบที่ต้องการ
            threshold (float): ค่าขีดแบ่งขั้นต่ำของคะแนนความคล้ายคลึง
            
        Returns:
            List[Dict[str, Any]]: รายการของคำตอบที่ดีที่สุดพร้อมคะแนน
        """
        # แปลงคำถามและคำตอบเป็น embeddings
        question_embedding = self.model.encode(question, convert_to_tensor=True)
        answer_embeddings = self.model.encode(candidate_answers, convert_to_tensor=True)
        
        # คำนวณความคล้ายคลึง
        cos_scores = F.cosine_similarity(question_embedding, answer_embeddings)
        
        # เรียงลำดับและเลือก top_k
        top_results = torch.topk(cos_scores, k=min(top_k, len(candidate_answers)))
        
        results = []
        for score, idx in zip(top_results.values, top_results.indices):
            score = score.item()
            if score >= threshold:
                results.append({
                    "answer": candidate_answers[idx],
                    "score": score,
                    "rank": len(results) + 1
                })
            
        return results
    
    def evaluate_qa(
        self, 
        questions: List[str], 
        correct_answers: List[str], 
        candidate_answers: List[List[str]]
    ) -> Dict[str, float]:
        """
        ประเมินผลโมเดลด้วยชุดข้อมูล QA
        
        Args:
            questions (List[str]): รายการของคำถาม
            correct_answers (List[str]): รายการของคำตอบที่ถูกต้อง
            candidate_answers (List[List[str]]): รายการของคำตอบที่เป็นไปได้สำหรับแต่ละคำถาม
            
        Returns:
            Dict[str, float]: ผลการประเมิน (accuracy@1, accuracy@5, MRR)
        """
        if not (len(questions) == len(correct_answers) == len(candidate_answers)):
            raise ValueError("จำนวนของ questions, correct_answers และ candidate_answers ต้องเท่ากัน")
            
        total = len(questions)
        acc_at_1 = 0
        acc_at_5 = 0
        mrr = 0.0
        
        for question, correct, candidates in zip(questions, correct_answers, candidate_answers):
            # ค้นหาคำตอบ
            results = self.find_answers(question, candidates, top_k=5)
            
            # ตรวจสอบ accuracy@1
            if results and results[0]["answer"] == correct:
                acc_at_1 += 1
                
            # ตรวจสอบ accuracy@5
            if any(r["answer"] == correct for r in results):
                acc_at_5 += 1
                
            # คำนวณ MRR
            for rank, result in enumerate(results, 1):
                if result["answer"] == correct:
                    mrr += 1.0 / rank
                    break
                    
        # คำนวณค่าเฉลี่ย
        metrics = {
            "accuracy@1": acc_at_1 / total,
            "accuracy@5": acc_at_5 / total,
            "mrr": mrr / total
        }
        
        return metrics 