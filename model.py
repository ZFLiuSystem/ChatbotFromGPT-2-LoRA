import torch.nn as nn
from transformers import GPT2Model
from peft import LoraConfig, TaskType, get_peft_model


class ChatModel(nn.Module):
    def __init__(self, model_path, embed_dim, vocab_size,
                 lora_r, lora_alpha, lora_dropout):
        super(ChatModel, self).__init__()
        gpt_backbone = GPT2Model.from_pretrained(model_path)
        self.gpt_backbone = self.lora_setup(gpt_backbone, lora_r, lora_alpha, lora_dropout)
        self.head = nn.Linear(in_features=embed_dim, out_features=vocab_size, bias=False)
        self.dropout = nn.Dropout(p=0.1)
        pass

    @staticmethod
    def lora_setup(model, lora_r, lora_alpha, lora_dropout):
        peft_config = LoraConfig(task_type=TaskType.FEATURE_EXTRACTION, inference_mode=False,
                                 r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
        lora_model = get_peft_model(model, peft_config=peft_config)
        return lora_model

    def forward(self, input_ids, attention_mask=None):
        hidden_states = self.gpt_backbone(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.head(hidden_states[0])
        logits = self.dropout(logits)
        return logits
        pass


