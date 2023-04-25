from ast import parse
from collections import defaultdict
from pathlib import Path, PosixPath
import re
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from kd_finetune import Args as KDFinetuneArgs
from finetune import Args as FinetuneArgs

DATASETS = ["QNLI","RTE","SST-2","MRPC","MNLI","QQP","CoLA"]
DEFAULT_METRIC = {
    "QNLI": "accuracy",
    "RTE": "accuracy",
    "SST-2": "accuracy",
    "MNLI": "accuracy",
    "MRPC": "F1_score",
    "QQP": "F1_score",
    "CoLA": "matthews",
}
def parse_command(command):
    if "python kd_finetune.py" in command:
        args = KDFinetuneArgs().parse_args(command.split(" ")[2:])
    if "python finetune.py" in command:
        args = FinetuneArgs().parse_args(command.split(" ")[2:])
    return eval(str(args))

class TrainingRunResult:
    def __init__(self,command):
        self.args = parse_command(command)
        self.intermediate_evals = []
        self.epoch_evals = []


def read_log(file):
    all_runs = {}
    current_run = None

    with open(file,"r") as f:
        while True:
            line = f.readline()
            if not line:
                break

            match = re.match("INFO:root:Step ([0-9]+):\n", line)
            if match is not None:
                step = int(match.group(1))
                if f.readline() == "INFO:root:Eval results:":
                    eval_res = re.match("INFO:root:(.+)\n", f.readline()).group(1)
                    eval_res = eval(eval_res)
                    current_run.intermediate_evals.append(eval_res | {"step":step})
            
            match = re.match("INFO:root:{\n", line)
            if match is not None:
                epoch_eval_lines = ""
                curr_line = "{"
                while curr_line != "}\n":
                    epoch_eval_lines += curr_line
                    curr_line = f.readline()
                epoch_eval_lines += "}"
                current_run.epoch_evals.append(eval(epoch_eval_lines))

            match = re.match("INFO:root:({'command_line'.+})\n", line)
            if match is not None:
                command = eval(match.group(1))['command_line']
                current_run = TrainingRunResult(command)
                all_runs[str(current_run.args)] = current_run
    return list(all_runs.values())

def get_metric(epoch_res,metric):
        if "dev_metrics" in epoch_res:
            return epoch_res["dev_metrics"][metric]
        else:
            return epoch_res["dev_accuracy"]

def get_metric(epoch_res,metric):
        if "dev_metrics" in epoch_res:
            return epoch_res["dev_metrics"][metric]
        else:
            return epoch_res["dev_accuracy"]

def to_dataframe(logfile, metric="accuracy"):
    training_runs = read_log(logfile)

    rows = []
    for run in training_runs:
        epoch_metrics = [get_metric(epoch_res, metric) for epoch_res in run.epoch_evals]
        
        rows.append({
            "lr":run.args["lr"],
            "batch_size":run.args["batch_size"],
            "best_metric":max(epoch_metrics),
            "last3_mean":np.mean(epoch_metrics[-3:]),
            "last3_std":np.std(epoch_metrics[-3:]),
        })
    
    return pd.DataFrame(rows)

def make_label(args,label_args):
    label = []
    for argname in label_args:
        label.append(f"{argname}={args[argname]}")
    return ",".join(label)

def make_plot(results : TrainingRunResult, label_args=["lr","batch_size"], metric="accuracy",):
    num_epochs = len(results.epoch_evals)
    metrics = []
    for epoch_res in results.epoch_evals:
        metrics.append(get_metric(epoch_res, metric))
    plt.plot(range(1,num_epochs+1), metrics,label=make_label(results.args,label_args))

def make_plots(logfile, metric="accuracy"):
    training_runs = read_log(logfile)
    done_runs = set()
    for run in training_runs:
        if str(run.args) in done_runs:
            continue
        make_plot(run, metric=metric)
        done_runs.add(str(run.args))
    plt.legend()
    plt.show()

if __name__=="__main__":
    read_log("../checkpoints/kd_finetune/tinybert/SST-2/tinybert/long_pretrain/prediction/log")