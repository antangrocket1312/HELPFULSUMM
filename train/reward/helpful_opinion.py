import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig


class HelpfulOpinionRegressor(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.device = torch.device("cpu")  # <<< force CPU

        self.config = AutoConfig.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name, config=self.config, low_cpu_mem_usage=True)
        self.dropout = nn.Dropout(
            self.config.hidden_dropout_prob if hasattr(self.config, "hidden_dropout_prob") else 0.1)
        self.head = nn.Linear(self.config.hidden_size, 1)  # scalar

        # Ensure whole module is on CPU (paranoia; everything should already be)
        self.to(self.device)
        self.eval()  # inference by default for a reward model

    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None):
        if input_ids.device.type != "cpu":
            input_ids = input_ids.to("cpu")
        if attention_mask is not None and attention_mask.device.type != "cpu":
            attention_mask = attention_mask.to("cpu")
        if token_type_ids is not None and token_type_ids.device.type != "cpu":
            token_type_ids = token_type_ids.to("cpu")

        out = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        # pool: prefer CLS token; if pooler exists (BERT), you can use pooler_output
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            pooled = out.pooler_output
        else:
            pooled = out.last_hidden_state[:, 0, :]  # CLS
        x = self.dropout(pooled)
        raw = self.head(x).squeeze(-1)  # shape [B]
        pred01 = torch.sigmoid(raw)  # in [0,1]
        pred = pred01 * 4.0 + 1.0  # bound to [1,5]
        outputs = {"logits": pred.unsqueeze(-1)}  # Trainer expects "logits"
        if labels is not None:
            labels = labels.to(pred.dtype)
            loss = F.mse_loss(pred, labels)
            outputs["loss"] = loss
        return outputs