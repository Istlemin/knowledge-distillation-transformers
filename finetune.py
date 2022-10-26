from distutils.command import check
from pathlib import Path
from typing import Optional
from datasets.dataset_dict import DatasetDict
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from transformers import get_scheduler
import time

from tqdm.auto import tqdm

def run_epoch(model,dataloader:DataLoader,device, optimizer:Optional[torch.optim.Optimizer]=None):
    progress_bar = tqdm(range(len(dataloader)))
    losses = []
    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        if optimizer is not None:
            optimizer.zero_grad()
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        losses.append(loss.detach().cpu())
        if optimizer is not None:
            optimizer.step()
        progress_bar.update(1)
    loss = torch.mean(torch.stack(losses))
    return loss

from datasets.arrow_dataset import Dataset
# def f(x:Dataset):
#     x.select()

def finetune(model : torch.nn.Module,tokenized_datasets : DatasetDict, lr=3e-5, checkpoint_path:Optional[Path]=None):
    train_dataloader = DataLoader(tokenized_datasets["train"], shuffle=True, batch_size=16)
    eval_dataloader = DataLoader(tokenized_datasets["dev"], shuffle=True, batch_size=16)

    optimizer = Adam(model.parameters(), lr=lr)
    num_epochs = 3
    num_training_steps = num_epochs * len(train_dataloader)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    if checkpoint_path is not None:
        checkpoint_path.mkdir(parents=True,exist_ok=True)

    model.train()
    for epoch in range(num_epochs):
        print(f"EPOCH {epoch}:")
        
        train_loss = run_epoch(model,train_dataloader,optimizer=optimizer,device=device)
        test_loss = run_epoch(model,eval_dataloader,device=device)
        print("Train loss:",train_loss)
        print("Test loss:",test_loss)

        if checkpoint_path is not None:
            print("Saving checkpoint...")
            torch.save({
                'epochs': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss
            }, checkpoint_path / time.strftime("%Y-%m-%d_%H:%M:%S"))