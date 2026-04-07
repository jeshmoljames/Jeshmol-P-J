import torch
from transformers import TimesformerModel

class VideoEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = TimesformerModel.from_pretrained(
            "facebook/timesformer-base-finetuned-k400"
        )

    def forward(self, x):
        return self.model(x).last_hidden_state.mean(dim=1)
