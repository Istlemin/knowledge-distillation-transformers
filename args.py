from pathlib import Path
from typing import List, Optional
from tap import Tap

class TrainArgs(Tap):
    seed: int = 0
    lr: float = 2e-5
    weight_decay : float = 0.01
    num_epochs: int
    batch_size: int = 64
    scheduler: Optional[str] = None
    port: Optional[int] = None
    num_gpus: int = 1

class FinetuneArgs(TrainArgs):
    gluepath: Path
    dataset:str
    outputdir:Path
    use_augmented_data:bool=False
    eval_steps: int=None
    metric: Optional[str] = None

class KDArgs(Tap):
    quantize:bool=False
    kd_losses:List[str]

if __name__=="__main__":
    class Args(KDArgs,FinetuneArgs):
        teacher_model:Path
        student_model:Path

    args = Args().parse_args()
    print(args)