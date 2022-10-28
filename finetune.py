from distutils.command import check
from pathlib import Path
from typing import Optional
from datasets.dataset_dict import DatasetDict
import torch
from torch.cuda import Device
from torch.utils.data import DataLoader
from torch.optim import Adam
from transformers import get_scheduler
import time

from tqdm.auto import tqdm

def run_epoch(model,dataloader:DataLoader,device:Device, optimizer:Optional[torch.optim.Optimizer]=None):
    if optimizer is None:
        model.eval()
    else:
        model.train()
    
    progress_bar = tqdm(range(len(dataloader)))
    losses = []
    correct_predictions = 0
    total_predictions = 0
    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        if optimizer is not None:
            optimizer.zero_grad()
        outputs = model(**batch)

        predictions = torch.argmax(outputs.logits,dim=1)
        correct_predictions += torch.sum(predictions==batch["labels"]).item()
        total_predictions += len(predictions)

        loss = torch.sum(outputs.loss)
        loss.backward()
        losses.append(loss.detach().cpu())
        if optimizer is not None:
            optimizer.step()
        progress_bar.update(1)
    loss = torch.mean(torch.stack(losses))
    return loss, correct_predictions/total_predictions

def finetune(model : torch.nn.Module,tokenized_datasets : DatasetDict, lr=3e-5, checkpoint_path:Optional[Path]=None, device_ids=None):
    train_dataloader = DataLoader(tokenized_datasets["train"], shuffle=True, batch_size=16)
    eval_dataloader = DataLoader(tokenized_datasets["dev"], shuffle=True, batch_size=16)

    optimizer = Adam(model.parameters(), lr=lr)
    num_epochs = 3
    
    device = torch.device("cpu")
    if device_ids is not None:
        if not torch.cuda.is_available():
            print("WARNING: [device_ids] was specified but CUDA is not available")
        else:
            device = torch.device("cuda")
            model = torch.nn.DataParallel(model, device_ids=device_ids)
            model.to(device)

    if checkpoint_path is not None:
        checkpoint_path.mkdir(parents=True,exist_ok=True)

    for epoch in range(num_epochs):
        print(f"EPOCH {epoch}:")

        model.train()        
        train_loss, train_accuracy = run_epoch(model,train_dataloader,optimizer=optimizer,device=device)
        model.eval()
        test_loss, test_accuracy = run_epoch(model,eval_dataloader,device=device)
        print("Train loss:",train_loss, "\t\tTrain accuracy:",train_accuracy)
        print("Test loss:",test_loss, "\t\tTest accuracy:",test_accuracy)
        
        if checkpoint_path is not None:
            print("Saving checkpoint...")

            if type(model) == torch.nn.DataParallel:
                model_state_dict = model.module.state_dict()
            else:
                model_state_dict = model.state_dict()

            torch.save({
                'epochs': epoch+1,
                'model_state_dict': model_state_dict,
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'test_loss': train_loss,
                'train_accuracy': train_accuracy,
                'test_accuracy': test_accuracy,
            }, checkpoint_path / time.strftime("%Y-%m-%d_%H:%M:%S"))