import os
import torch
import argparse
import importlib
import numpy as np
import pytorch_lightning as pl
from models.clipmodel import CLIPModel
from transformers import CLIPConfig, BertConfig, BertModel
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def _import_class(module_and_class_name: str) -> type:
    
    module_name, class_name = module_and_class_name.rsplit(".", 1)
    module = importlib.import_module(module_name)
    class_ = getattr(module, class_name)
    return class_


def _setup_parser(): 
    
    parser = argparse.ArgumentParser(add_help=False)

    
    trainer_parser = pl.Trainer.add_argparse_args(parser)
    trainer_parser._action_groups[1].title = "Trainer Args"  
    parser = argparse.ArgumentParser(add_help=False, parents=[trainer_parser])

    parser.add_argument("--model", type=str, default="RobertaUseLabelWord")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--task_name", type=str, default=None)
    parser.add_argument("--wandb", action="store_true", default=False)
    parser.add_argument("--Lmodel_class", type=str, default="TransformerLmodel")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("data", type=str, default="MKGC")
    parser.add_argument("--chunk", type=str, default="")
    temp_args, _ = parser.parse_known_args()
    data_class = _import_class(f"data.{temp_args.data_class}")
    model_class = _import_class(f"models.{temp_args.model_class}")
    lit_model_class = _import_class(f"lmodels.{temp_args.Lmodel_class}")
    data_group = parser.add_argument_group("Data Args")
    data_class.add_to_argparse(data_group)

    model_group = parser.add_argument_group("Model Args")
    if hasattr(model_class, "add_to_argparse"):
        model_class.add_to_argparse(model_group)

    lit_model_group = parser.add_argument_group("Lmodel Args")
    lit_model_class.add_to_argparse(lit_model_group)

    parser.add_argument("--help", "-h", action="help")
    return parser


def main():
    parser = _setup_parser()
    args = parser.parse_args()
    print(args)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    pl.seed_everything(args.seed)

    data_class = _import_class(f"data.{args.data_class}")               
    model_class = _import_class(f"models.{args.model_class}")           
    Lmodel_class = _import_class(f"lmodels.{args.Lmodel_class}") 

    vision_config = CLIPConfig.from_pretrained('openai/clip-vit-base-patch32').vision_config
    text_config = BertConfig.from_pretrained('bert-base-uncased')
    bert = BertModel.from_pretrained('bert-base-uncased')
    clip_model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
    clip_vit = clip_model.vision_model
    
    vision_config.device = 'cpu'
    model = model_class(vision_config, text_config)
    clip_model_dict = clip_vit.state_dict()
    text_model_dict = bert.state_dict()

    def load_state_dict():
        
        vision_names, text_names = [], []
        model_dict = model.state_dict()
        for name in model_dict:
            if 'vision' in name:
                clip_name = name.replace('vision_', '').replace('model.', '').replace('interation.', '')
                if clip_name in clip_model_dict:
                    vision_names.append(clip_name)
                    model_dict[name] = clip_model_dict[clip_name]
            elif 'text' in name:
                text_name = name.replace('text_', '').replace('model.', '').replace('interation.', '')
                if text_name in text_model_dict:
                    text_names.append(text_name)
                    model_dict[name] = text_model_dict[text_name]
        assert len(vision_names) == len(clip_model_dict) and len(text_names) == len(text_model_dict), \
                    (len(vision_names), len(text_names), len(clip_model_dict), len(text_model_dict))
        model.load_state_dict(model_dict)
        print('Load model successful.')
    load_state_dict()

    data = data_class(args, model)
    tokenizer = data.tokenizer

    lit_model = Lmodel_class(args=args, model=model, tokenizer=tokenizer, data_config=data.get_config()) 
    if args.checkpoint: 
        lit_model.load_state_dict(torch.load(args.checkpoint, map_location="cpu")["state_dict"])

    logger = pl.loggers.TensorBoardLogger("result/logs") 
    if args.wandb:
        logger = pl.loggers.WandbLogger(project="MKGC_bert", name=args.data_dir.split("/")[-1])
        logger.log_hyperparams(vars(args))

    metric_name = "Eval/hits10"

    early_callback = pl.callbacks.EarlyStopping(monitor="Eval/mrr", mode="max", patience=5) 
    model_checkpoint = pl.callbacks.ModelCheckpoint(monitor=metric_name, mode="max",
        filename=args.data_dir.split("/")[-1] + '/{epoch}' if not args.pretrain else args.data_dir.split("/")[-1] ,
        dirpath="result",
        save_weights_only=True,
    )
    callbacks = [early_callback, model_checkpoint]

    trainer = pl.Trainer.from_argparse_args(args,   
                                            callbacks=callbacks, 
                                            logger=logger, 
                                            default_root_dir="result/logs",)
    
    if "EntityEmbedding" not in lit_model.__class__.__name__:
        trainer.fit(lit_model, datamodule=data) 
        path = model_checkpoint.best_model_path 
        lit_model.load_state_dict(torch.load(path)["state_dict"])

    result = trainer.test(lit_model, datamodule=data) 
    print(result)

   
    if "EntityEmbedding" not in lit_model.__class__.__name__:
        print("*path"*20)
        print(path)





if __name__ == "__main__": 

    main()
