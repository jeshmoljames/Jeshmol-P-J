import torch
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from models.timesformer_model import VideoEncoder

# Load config
with open("configs/config.yaml") as f:
    config = yaml.safe_load(f)

device = "cuda" if torch.cuda.is_available() else "cpu"

model = VideoEncoder().to(device)

optimizer = AdamW(
    model.parameters(),
    lr=config["training"]["learning_rate"]
)

scheduler = CosineAnnealingLR(
    optimizer,
    T_max=config["training"]["epochs"]
)

criterion = torch.nn.CrossEntropyLoss()

# Dummy dataset placeholder
train_loader = []  # replace with actual dataloader

best_loss = float("inf")
patience = config["training"]["early_stopping"]
counter = 0

for epoch in range(config["training"]["epochs"]):
    model.train()
    total_loss = 0

    for videos, labels in train_loader:
        videos, labels = videos.to(device), labels.to(device)

        outputs = model(videos)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    scheduler.step()

    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

    # Early stopping
    if total_loss < best_loss:
        best_loss = total_loss
        counter = 0
        torch.save(model.state_dict(), "best_model.pth")
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping triggered")
            break
