"""
SOMI Multi-Teacher Distillation Pipeline
==========================================

Transfers knowledge from one or more teacher models (e.g., SOMI-wrapped
Llama, Mistral, Phi) into the pure SOMI-SSM language model.

The process:
1. Run teacher on a text corpus to get soft output distributions
2. Train student (pure SOMI) to match those distributions via KL divergence
3. Student's SOMI layers also do local learning (stress/geometry) simultaneously
4. If stress is high, auto-growth adds neurons

Supports multiple teachers with different specialties:
- General teacher -> all layers
- Code teacher -> specific layers
- Math teacher -> specific layers

Usage:
    student = SOMILanguageModel(vocab_size=32000, hidden_dim=256)
    distiller = Distiller(student, temperature=2.0, alpha=0.5)
    distiller.distill_from_teacher(teacher_model, tokenizer, text_data)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Callable
from .growth import AutoGrowth


class Distiller:
    """
    Distills knowledge from teacher model(s) into a pure SOMI student.

    The loss function:
        L = alpha * KL(teacher_soft || student_soft) + (1-alpha) * CE(student, labels)

    Where:
        - KL is the Kullback-Leibler divergence on temperature-scaled logits
        - CE is standard cross-entropy on hard labels
        - alpha controls the balance (0.5 = equal weight)
        - temperature softens the distributions (higher = smoother)

    Args:
        student: the pure SOMILanguageModel
        temperature: softening temperature for KL divergence
        alpha: weight for distillation loss vs hard label loss
        lr: learning rate for student's macro parameters (embeddings, LM head)
        auto_grow: whether to enable stress-triggered growth
        max_hidden: maximum hidden dim for auto-growth
    """

    def __init__(
        self,
        student: 'SOMILanguageModel',
        temperature: float = 2.0,
        alpha: float = 0.5,
        lr: float = 1e-3,
        auto_grow: bool = True,
        max_hidden: int = 1024,
    ):
        self.student = student
        self.temperature = temperature
        self.alpha = alpha

        self.optimizer = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=0.01)

        self.growth_monitor = None
        if auto_grow:
            self.growth_monitor = AutoGrowth(
                student, stress_threshold=0.5, patience=50,
                growth_factor=1.5, max_hidden=max_hidden,
            )

        self.step_count = 0
        self.loss_history = []

    def distillation_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute the combined distillation + hard label loss.

        Args:
            student_logits: [batch, seq, vocab] from student
            teacher_logits: [batch, seq, vocab] from teacher
            labels: [batch, seq] hard labels (optional)

        Returns:
            loss: scalar
        """
        T = self.temperature

        # Align vocab sizes if they differ (teacher may have larger vocab)
        if student_logits.shape[-1] != teacher_logits.shape[-1]:
            min_vocab = min(student_logits.shape[-1], teacher_logits.shape[-1])
            student_logits = student_logits[..., :min_vocab]
            teacher_logits = teacher_logits[..., :min_vocab]

        student_soft = F.log_softmax(student_logits / T, dim=-1)
        teacher_soft = F.softmax(teacher_logits / T, dim=-1)

        kl_loss = F.kl_div(
            student_soft.view(-1, student_soft.shape[-1]),
            teacher_soft.view(-1, teacher_soft.shape[-1]),
            reduction='batchmean',
        ) * (T * T)

        if labels is not None:
            shift_logits = student_logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            ce_loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.shape[-1]),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            loss = self.alpha * kl_loss + (1 - self.alpha) * ce_loss
        else:
            loss = kl_loss

        return loss

    @torch.no_grad()
    def _get_teacher_logits(self, teacher, input_ids):
        """Get soft labels from teacher model."""
        teacher.eval()
        outputs = teacher(input_ids)
        if isinstance(outputs, torch.Tensor):
            return outputs
        elif hasattr(outputs, 'logits'):
            return outputs.logits
        elif isinstance(outputs, tuple):
            return outputs[0]
        else:
            return outputs[0]

    def distill_step(
        self,
        teacher,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        One distillation step.

        Args:
            teacher: the teacher model (any model that produces logits)
            input_ids: [batch, seq] token IDs
            labels: [batch, seq] target IDs (optional)

        Returns:
            info dict with loss, growth status, etc.
        """
        teacher_logits = self._get_teacher_logits(teacher, input_ids)

        self.student.train()
        student_logits, _, diagnostics = self.student(input_ids, training=True)

        loss = self.distillation_loss(student_logits, teacher_logits, labels)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.student.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.step_count += 1
        self.loss_history.append(loss.item())

        grew = False
        if self.growth_monitor:
            grew = self.growth_monitor.step(diagnostics)
            if grew:
                self.optimizer = torch.optim.AdamW(
                    self.student.parameters(), lr=self.optimizer.defaults['lr'],
                    weight_decay=0.01,
                )

        return {
            'step': self.step_count,
            'loss': loss.item(),
            'kl_component': loss.item(),
            'grew': grew,
            'hidden_dim': self.student.hidden_dim,
        }

    def distill_from_teacher(
        self,
        teacher,
        tokenizer,
        texts: List[str],
        max_length: int = 128,
        n_epochs: int = 1,
        log_every: int = 10,
    ) -> dict:
        """
        Full distillation loop from a single teacher.

        Args:
            teacher: HuggingFace model or SOMI-wrapped model
            tokenizer: the tokenizer
            texts: list of text strings to distill on
            max_length: maximum sequence length
            n_epochs: number of passes through the data
            log_every: print progress every N steps

        Returns:
            summary dict
        """
        device = next(self.student.parameters()).device

        print(f"\nDistilling from teacher ({len(texts)} texts, {n_epochs} epochs)...")
        print(f"  Student: H={self.student.hidden_dim}, {self.student.n_layers} layers")

        for epoch in range(n_epochs):
            epoch_loss = 0.0
            for i, text in enumerate(texts):
                tokens = tokenizer(
                    text, return_tensors='pt', max_length=max_length,
                    truncation=True, padding='max_length',
                ).to(device)

                input_ids = tokens['input_ids']
                info = self.distill_step(teacher, input_ids, labels=input_ids)
                epoch_loss += info['loss']

                if (i + 1) % log_every == 0:
                    avg = epoch_loss / (i + 1)
                    print(f"  Epoch {epoch+1} Step {i+1}/{len(texts)} | Loss: {avg:.4f} | H={info['hidden_dim']}")

            avg_epoch_loss = epoch_loss / max(len(texts), 1)
            print(f"  Epoch {epoch+1} complete | Avg loss: {avg_epoch_loss:.4f}")

        return {
            'final_loss': self.loss_history[-1] if self.loss_history else 0,
            'total_steps': self.step_count,
            'final_hidden_dim': self.student.hidden_dim,
            'growth_events': self.growth_monitor.growth_events if self.growth_monitor else [],
        }
