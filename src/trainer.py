import torch
import torch.nn as nn
from transformers import Trainer


# reference: https://www.kaggle.com/c/tweet-sentiment-extraction/discussion/143764
class FGM:
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1.0, emb_name="word_embeddings"):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0:
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name="word_embeddings"):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
            self.backup = {}


class TreasureFGMTrainer(Trainer):
    def __init__(self, adv_start_epoch, adv_epsilon, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.adv_start_epoch = adv_start_epoch
        self.adv_epsilon = adv_epsilon
        self.fgm = FGM(self.model)
        self.loss_fn = nn.BCEWithLogitsLoss()

    def manual_backword(self, loss):
        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
            # deepspeed handles loss scaling by gradient_accumulation_steps in its `backward`
            loss = loss / self.args.gradient_accumulation_steps

        self.accelerator.backward(loss)

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels").float()
        _inputs = {k: v for k, v in inputs.items() if k != "labels"}

        outputs = model(**_inputs)
        logits = outputs.get("logits")
        loss = self.loss_fn(logits.view(-1), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

    def training_step(self, model, inputs) -> torch.Tensor:
        model.train()
        inputs = self._prepare_inputs(inputs)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)
        self.manual_backword(loss)

        # adversarial training
        current_epoch = self.state.epoch
        if current_epoch >= self.adv_start_epoch:
            self.fgm.attack(epsilon=self.adv_epsilon)
            with self.compute_loss_context_manager():
                loss_adv = self.compute_loss(model, inputs)
            self.manual_backword(loss_adv)
            self.fgm.restore()

        return loss.detach()
