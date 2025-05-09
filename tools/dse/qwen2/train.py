import os
import sys
import numpy as np

from transformers import AutoProcessor
from transformers.trainer_callback import TrainerCallback
from transformers import (
    HfArgumentParser,
    set_seed,
)
from trainer import TevatronTrainer as Trainer

from arguments import (
    ModelArguments,
    DataArguments,
    TevatronTrainingArguments as TrainingArguments,
)
from dse import DSEModel
from dataset import TrainDataset
from custom_dataset import CustomDataset
from collator import TrainCollator
import visdom


class LoggingCallback(TrainerCallback):
    # def on_substep_end(self, args, state, control, **kwargs):
    #     print("inside on substep end callback")
    #     if len(state.log_history) > 0:
    #         print("state", state.log_history[-1]["loss"])
    def __init__(self):
        self.vis = visdom.Visdom(env="dse", log_to_filename="/dev/null")
        self.losses = []

    # def on_step_end(self, args, state, control, **kwargs):
    #     print("state ", state)
    #     print("args ", args)
    #     print("control ", control)
    #     if len(state.log_history) > 0:
    #         self.losses.append(state.log_history[-1]["loss"])
    #         # print("self.losses ", self.losses)
    #         X = list(range(1, len(self.losses) + 1))
    #         Y_list = self.losses
    #         opts = {"legend": ["dse loss"]}
    #         self.vis.line(X=X, Y=np.array(Y_list).T, win="loss", opts=opts)
    def on_log(self, args, state, control, **kwargs):
        try:
            self.losses.append(kwargs["logs"]["loss"])
            X = list(range(1, len(self.losses) + 1))
            Y_list = self.losses
            opts = {"legend": ["dse loss"]}
            self.vis.line(X=X, Y=np.array(Y_list).T, win="loss", opts=opts)
        except Exception:
            pass


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
        model_args: ModelArguments
        data_args: DataArguments
        training_args: TrainingArguments

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )
    set_seed(training_args.seed)
    processor = AutoProcessor.from_pretrained(
        model_args.tokenizer_name
        if model_args.tokenizer_name
        else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        trust_remote_code=True,
        min_pixels=28 * 28,
        max_pixels=1280 * 28 * 28,
    )
    processor.tokenizer.padding_side = "left"

    model = DSEModel.build(
        model_args,
        training_args,
        cache_dir=model_args.cache_dir,
    )

    if data_args.dataset_name == "custom":
        train_dataset = CustomDataset(
            data_args.dataset_path,
            data_args.corpus_path,
            data_args.train_group_size - 1,
        )
    else:
        train_dataset = TrainDataset(data_args)
    collator = TrainCollator(data_args, processor)

    trainer_cls = Trainer
    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
        callbacks=[LoggingCallback],
    )
    train_dataset.trainer = trainer
    trainer.train()
    trainer.save_model()
    if trainer.is_world_process_zero():
        processor.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()
