"""
โมดูลสำหรับการปรับแต่งโมเดลด้วยเทคนิค DPO (Direct Preference Optimization)
"""

import os
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Union, Any, Tuple
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    Trainer, 
    TrainingArguments,
    PreTrainedModel
)
from peft import get_peft_model, LoraConfig, TaskType
from datasets import Dataset

class LLMDPOTrainer:
    """คลาสสำหรับการปรับแต่งโมเดลด้วยเทคนิค DPO"""
    
    def __init__(
        self, 
        model_name: str, 
        tokenizer_name: Optional[str] = None, 
        device: Optional[str] = None,
        beta: float = 0.1,
        use_peft: bool = False,
        peft_config: Optional[Dict[str, Any]] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        tokenizer_kwargs: Optional[Dict[str, Any]] = None
    ):
        """
        กำหนดค่าเริ่มต้นสำหรับการปรับแต่งโมเดล
        
        Args:
            model_name (str): ชื่อหรือพาธของโมเดลที่จะทำการปรับแต่ง
            tokenizer_name (str, optional): ชื่อหรือพาธของ tokenizer ถ้าไม่ระบุจะใช้ค่าเดียวกับ model_name
            device (str, optional): อุปกรณ์ที่ใช้ในการคำนวณ ('cuda', 'cpu') ถ้าไม่ระบุจะเลือกอัตโนมัติ
            beta (float): พารามิเตอร์ beta สำหรับ DPO (ค่าเริ่มต้น 0.1)
            use_peft (bool): ใช้ PEFT (Parameter-Efficient Fine-Tuning) หรือไม่
            peft_config (Dict, optional): การตั้งค่าสำหรับ PEFT
            model_kwargs (Dict, optional): พารามิเตอร์เพิ่มเติมสำหรับการโหลดโมเดล
            tokenizer_kwargs (Dict, optional): พารามิเตอร์เพิ่มเติมสำหรับการโหลด tokenizer
        """
        self.model_name = model_name
        self.tokenizer_name = tokenizer_name if tokenizer_name else model_name
        self.beta = beta
        self.use_peft = use_peft
        
        # กำหนดค่าเริ่มต้นถ้าไม่ได้ระบุ
        if model_kwargs is None:
            model_kwargs = {}
        if tokenizer_kwargs is None:
            tokenizer_kwargs = {}
        
        # กำหนดอุปกรณ์ที่ใช้ในการคำนวณ
        if device:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
        # โหลด tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_name, 
            padding_side="right",
            **tokenizer_kwargs
        )
        
        # ตรวจสอบว่ามี pad_token หรือไม่
        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                self.tokenizer.pad_token = self.tokenizer.eos_token = "</s>"
        
        # โหลดโมเดล
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map=self.device if self.device != "cpu" else None,
            **model_kwargs
        )
        
        # ใช้ PEFT ถ้าต้องการ
        if self.use_peft:
            if peft_config is None:
                # ค่าเริ่มต้นสำหรับ LoRA
                peft_config = {
                    "r": 16,
                    "lora_alpha": 32,
                    "lora_dropout": 0.05,
                    "bias": "none",
                    "task_type": TaskType.CAUSAL_LM
                }
                
            lora_config = LoraConfig(**peft_config)
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()
        
        # ย้ายโมเดลไปยังอุปกรณ์
        self.model.to(self.device)
        
        # ตั้งค่าโมเดลเป็นโหมดเทรน
        self.model.train()
        
    def prepare_dataset(
        self, 
        prompts: List[str], 
        chosen_responses: List[str], 
        rejected_responses: List[str], 
        max_length: int = 512
    ) -> Dataset:
        """
        เตรียมชุดข้อมูลสำหรับการเทรน DPO
        
        Args:
            prompts (List[str]): รายการของ prompt
            chosen_responses (List[str]): รายการของคำตอบที่ถูกเลือก (preferred)
            rejected_responses (List[str]): รายการของคำตอบที่ถูกปฏิเสธ (less preferred)
            max_length (int): ความยาวสูงสุดของข้อความ
            
        Returns:
            Dataset: ชุดข้อมูลที่เตรียมพร้อมแล้ว
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
            # สร้างคู่ prompt-response สำหรับคำตอบที่ถูกเลือกและถูกปฏิเสธ
            chosen_texts = [prompt + chosen for prompt, chosen in zip(examples["prompt"], examples["chosen"])]
            rejected_texts = [prompt + rejected for prompt, rejected in zip(examples["prompt"], examples["rejected"])]
            
            # Tokenize ข้อความ
            chosen_tokens = self.tokenizer(
                chosen_texts, 
                padding="max_length", 
                truncation=True, 
                max_length=max_length, 
                return_tensors="pt"
            )
            
            rejected_tokens = self.tokenizer(
                rejected_texts, 
                padding="max_length", 
                truncation=True, 
                max_length=max_length, 
                return_tensors="pt"
            )
            
            # Tokenize เฉพาะ prompt เพื่อสร้าง mask
            prompt_tokens = self.tokenizer(
                examples["prompt"], 
                padding="max_length", 
                truncation=True, 
                max_length=max_length, 
                return_tensors="pt"
            )
            
            # สร้าง mask สำหรับแยกระหว่าง prompt และ response
            prompt_masks = []
            for i in range(len(prompt_tokens["input_ids"])):
                # หาความยาวของ prompt ที่แท้จริง (ไม่รวม padding)
                prompt_len = (prompt_tokens["attention_mask"][i] == 1).sum().item()
                
                # สร้าง mask โดยตั้งค่าให้ 0 สำหรับ prompt และ 1 สำหรับ response
                # 0 = prompt, 1 = response
                mask = [0] * prompt_len + [1] * (max_length - prompt_len)
                mask = mask[:max_length]  # ตัดให้มีความยาวไม่เกิน max_length
                prompt_masks.append(mask)
            
            return {
                "chosen_input_ids": chosen_tokens["input_ids"],
                "chosen_attention_mask": chosen_tokens["attention_mask"],
                "rejected_input_ids": rejected_tokens["input_ids"],
                "rejected_attention_mask": rejected_tokens["attention_mask"],
                "prompt_mask": torch.tensor(prompt_masks)
            }
        
        # แปลงข้อมูลด้วย tokenize_function
        tokenized_dataset = dataset.map(
            tokenize_function, 
            batched=True,
            remove_columns=dataset.column_names
        )
        
        # กำหนดรูปแบบข้อมูล
        tokenized_dataset.set_format(type="torch")
            
        return tokenized_dataset
    
    def _get_batch_logps(
        self, 
        model: PreTrainedModel, 
        input_ids: torch.LongTensor, 
        attention_mask: torch.Tensor, 
        labels: Optional[torch.LongTensor] = None
    ) -> torch.Tensor:
        """
        คำนวณ log probabilities สำหรับแต่ละโทเค็นในแบตช์
        
        Args:
            model: โมเดลภาษา
            input_ids: รหัสของโทเค็นอินพุต
            attention_mask: attention mask สำหรับอินพุต
            labels: โทเค็นเป้าหมาย (ถ้าไม่ระบุจะใช้ input_ids)
            
        Returns:
            torch.Tensor: log probabilities สำหรับแต่ละโทเค็น
        """
        if labels is None:
            labels = input_ids
            
        # คำนวณ logits
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        
        # เลื่อน logits ให้ตรงกับป้ายกำกับ (เนื่องจากต้องทำนายโทเค็นถัดไป)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        shift_attention_mask = attention_mask[:, 1:].contiguous()
        
        # คำนวณ log probabilities
        log_probs = F.log_softmax(shift_logits, dim=-1)
        
        # ดึง log probabilities ของป้ายกำกับที่ถูกต้อง
        selected_log_probs = log_probs.gather(
            dim=-1, 
            index=shift_labels.unsqueeze(-1)
        ).squeeze(-1)
        
        # คูณด้วย attention mask เพื่อเลือกเฉพาะโทเค็นที่ไม่ใช่ padding
        selected_log_probs = selected_log_probs * shift_attention_mask
        
        return selected_log_probs
    
    def _dpo_loss(
        self, 
        policy_chosen_logps: torch.Tensor, 
        policy_rejected_logps: torch.Tensor, 
        prompt_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        คำนวณ DPO loss
        
        Args:
            policy_chosen_logps: log probabilities ของคำตอบที่ถูกเลือกจากโมเดลปัจจุบัน
            policy_rejected_logps: log probabilities ของคำตอบที่ถูกปฏิเสธจากโมเดลปัจจุบัน
            prompt_mask: mask สำหรับแยกส่วน prompt ออกจากคำตอบ
            
        Returns:
            torch.Tensor: ค่า loss ที่คำนวณได้
        """
        # แยกเฉพาะส่วนของคำตอบ (ไม่รวม prompt) ในการคำนวณ loss
        response_chosen_logps = policy_chosen_logps.masked_select(prompt_mask[:, :-1].bool())
        response_rejected_logps = policy_rejected_logps.masked_select(prompt_mask[:, :-1].bool())
        
        # รวม log probabilities สำหรับแต่ละตัวอย่าง
        chosen_rewards = response_chosen_logps.sum()
        rejected_rewards = response_rejected_logps.sum()
        
        # คำนวณ DPO loss: -log(sigmoid(beta * (r_w(x_w) - r_w(x_l))))
        # ในที่นี้ r_w(x) คือ log probabilities ของคำตอบ
        logits = self.beta * (chosen_rewards - rejected_rewards)
        loss = -F.logsigmoid(logits).mean()
        
        return loss
    
    def train(
        self, 
        dataset: Dataset, 
        output_dir: str = "./results/dpo",
        num_train_epochs: int = 3,
        per_device_train_batch_size: int = 4,
        gradient_accumulation_steps: int = 1,
        learning_rate: float = 1e-5,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.1,
        save_steps: int = 500,
        eval_dataset: Optional[Dataset] = None,
        fp16: bool = True,
        **kwargs
    ) -> Trainer:
        """
        เทรนโมเดลด้วยเทคนิค DPO
        
        Args:
            dataset: ชุดข้อมูลที่เตรียมพร้อมแล้ว
            output_dir: ไดเรกทอรีสำหรับบันทึกผลลัพธ์
            num_train_epochs: จำนวนรอบการเทรน
            per_device_train_batch_size: ขนาดแบตช์ต่ออุปกรณ์
            gradient_accumulation_steps: จำนวนขั้นตอนสำหรับการสะสมเกรเดียนต์
            learning_rate: อัตราการเรียนรู้
            weight_decay: ค่าการลดน้ำหนัก
            warmup_ratio: สัดส่วนของขั้นตอนทั้งหมดที่ใช้สำหรับวอร์มอัป
            save_steps: บันทึกโมเดลทุกกี่ขั้นตอน
            eval_dataset: ชุดข้อมูลสำหรับการประเมินผล (ถ้ามี)
            fp16: ใช้การคำนวณแบบ 16-bit floating point หรือไม่
            **kwargs: พารามิเตอร์เพิ่มเติมสำหรับ TrainingArguments
            
        Returns:
            Trainer: ออบเจ็กต์ Trainer ที่ใช้ในการเทรน
        """
        # สร้างไดเรกทอรีเก็บผลลัพธ์ถ้ายังไม่มี
        os.makedirs(output_dir, exist_ok=True)
        
        # กำหนดฟังก์ชัน DPO loss สำหรับการเทรน
        def compute_loss(model, inputs):
            # แยกข้อมูลออกจาก inputs
            chosen_input_ids = inputs["chosen_input_ids"].to(self.device)
            chosen_attention_mask = inputs["chosen_attention_mask"].to(self.device)
            rejected_input_ids = inputs["rejected_input_ids"].to(self.device)
            rejected_attention_mask = inputs["rejected_attention_mask"].to(self.device)
            prompt_mask = inputs["prompt_mask"].to(self.device)
            
            # คำนวณ log probabilities สำหรับคำตอบที่ถูกเลือกและถูกปฏิเสธ
            policy_chosen_logps = self._get_batch_logps(
                model, 
                chosen_input_ids, 
                chosen_attention_mask
            )
            
            policy_rejected_logps = self._get_batch_logps(
                model, 
                rejected_input_ids, 
                rejected_attention_mask
            )
            
            # คำนวณ DPO loss
            loss = self._dpo_loss(
                policy_chosen_logps, 
                policy_rejected_logps, 
                prompt_mask
            )
            
            return loss
        
        # กำหนดค่าพารามิเตอร์สำหรับการเทรน
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            warmup_ratio=warmup_ratio,
            save_steps=save_steps,
            evaluation_strategy="steps" if eval_dataset is not None else "no",
            fp16=fp16 and self.device != "cpu",
            **kwargs
        )
        
        # สร้าง Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            eval_dataset=eval_dataset,
            compute_loss=compute_loss,
            tokenizer=self.tokenizer
        )
        
        # เริ่มการเทรน
        trainer.train()
        
        # บันทึกโมเดลและ tokenizer
        self.save_model(output_dir)
        
        return trainer
    
    def save_model(self, path: str):
        """
        บันทึกโมเดลและ tokenizer
        
        Args:
            path (str): พาธที่จะบันทึกโมเดล
        """
        os.makedirs(path, exist_ok=True)
        
        # ถ้าใช้ PEFT ให้บันทึกเฉพาะ adapter
        if self.use_peft:
            self.model.save_pretrained(path)
        else:
            self.model.save_pretrained(path)
            
        self.tokenizer.save_pretrained(path)
    
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
        # ตั้งค่าโมเดลเป็นโหมดประเมินผล
        self.model.eval()
        
        # Tokenize prompt
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
        generated_texts = [
            self.tokenizer.decode(output, skip_special_tokens=True) 
            for output in outputs
        ]
        
        # ตั้งค่าโมเดลกลับเป็นโหมดเทรน
        self.model.train()
        
        return generated_texts 