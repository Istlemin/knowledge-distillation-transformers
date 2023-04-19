from ast import parse
from collections import defaultdict
from pathlib import Path, PosixPath
import re
import json


from kd_finetune import Args as KDFinetuneArgs
from finetune import Args as FinetuneArgs

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

if __name__=="__main__":
    read_log("../checkpoints/kd_finetune/tinybert/SST-2/tinybert/long_pretrain/prediction/log")