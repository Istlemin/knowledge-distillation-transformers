import argparse
from transformers import AutoModelForPreTraining
from src.modeling.models import get_bert_config
from pathlib import Path
import torch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", dest="checkpoint_path", type=Path, required=True)
    parser.add_argument("--type", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--model_save_path", type=Path, required=True)
    args = parser.parse_args()

    model = AutoModelForPreTraining.from_config(get_bert_config(args.model))

    if args.type=="kd_pretrain":
        checkpoint = torch.load(args.checkpoint_path, map_location="cpu")

        state_dict = {k.removeprefix("module.student."):v for k,v in checkpoint["model_state_dict"].items() if "teacher" not in k}

        
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        assert len(missing_keys) + len(unexpected_keys) < 20, f"{missing_keys}, {unexpected_keys}"


    model.save_pretrained(args.model_save_path)

if __name__=="__main__":
    main()