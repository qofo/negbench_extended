import logging
import os
import re
import sys
import random

import numpy as np
import torch

try:
    import wandb
except ImportError:
    wandb = None

try:
    import torch.utils.tensorboard as tensorboard
except ImportError:
    tensorboard = None

from open_clip import create_model_and_transforms, get_tokenizer, create_model_from_pretrained
from training.data import get_data
from training.distributed import is_master, init_distributed_device
from training.logger import setup_logging
from training.params import parse_args

from src.evaluation.utils import evaluate, evaluate_video

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_descriptor')

LATEST_CHECKPOINT_NAME = "epoch_latest.pt"

def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)


def natural_key(string_):
    """See http://www.codinghorror.com/blog/archives/001018.html"""
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_.lower())]

def main(args):
    args = parse_args(args)

    if torch.cuda.is_available():
        # This enables tf32 on Ampere GPUs which is only 8% slower than
        # float16 and almost as accurate as float32
        # This was a default in pytorch until 1.12
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    # fully initialize distributed device environment
    device = init_distributed_device(args)

    if args.name is None:
        # Terminate session with error message
        raise ValueError("Please provide a name for the evaluation run")
        return -1
    
    # Set up logging
    log_base_path = os.path.join(args.logs, args.name)
    args.log_path = None
    if is_master(args, local=args.log_local):
        os.makedirs(log_base_path, exist_ok=True)
        log_filename = f'out-{args.rank}' if args.log_local else 'out.log'
        args.log_path = os.path.join(log_base_path, log_filename)

    # Setup text logger
    args.log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(args.log_path, args.log_level)

    # Setup wandb, tensorboard, checkpoint logging
    args.wandb = 'wandb' in args.report_to or 'all' in args.report_to
    args.tensorboard = 'tensorboard' in args.report_to or 'all' in args.report_to
    args.checkpoint_path = os.path.join(log_base_path, "checkpoints")
    if is_master(args):
        args.tensorboard_path = os.path.join(log_base_path, "tensorboard") if args.tensorboard else ''
        for dirname in [args.tensorboard_path, args.checkpoint_path]:
            if dirname:
                os.makedirs(dirname, exist_ok=True)
    else:
        args.tensorboard_path = ''

    # Load model
    if isinstance(args.force_image_size, (tuple, list)) and len(args.force_image_size) == 1:
        # arg is nargs, single (square) image size list -> int
        args.force_image_size = args.force_image_size[0]
    random_seed(args.seed, 0)
    model_kwargs = {}
    if args.siglip:
        model_kwargs['init_logit_scale'] = np.log(10)  # different from CLIP
        model_kwargs['init_logit_bias'] = -10

    if args.cxr_dataset and "biomedclip" in args.name:
        try:
            print("Loading model from pretrained: BioMedCLIP-PubMedBERT_256-vit_base_patch16_224")
            model, preprocess = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224', device=device)
            preprocess_train = preprocess_val = preprocess
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            return -1
    elif "quiltnet" in args.name:
        try:
            print("Loading model from pretrained: QuiltNet")
            model, preprocess = create_model_from_pretrained('hf-hub:wisdomik/QuiltNet-B-32', device=device)
            preprocess_train = preprocess_val = preprocess
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            return -1
    elif "conch" in args.name:
        try:
            print("Loading model from pretrained: ConCH")
            from conch.open_clip_custom import create_model_from_pretrained as create_model_from_pretrained_conch
            model, preprocess = create_model_from_pretrained_conch('conch_ViT-B-16', "hf_hub:MahmoodLab/conch", device=device, hf_auth_token="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")
            preprocess_train = preprocess_val = preprocess
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            return -1
    else:
        model, preprocess_train, preprocess_val = create_model_and_transforms(
            args.model,
            args.pretrained,
            precision=args.precision,
            device=device,
            jit=args.torchscript,
            force_quick_gelu=args.force_quick_gelu,
            force_custom_text=args.force_custom_text,
            force_patch_dropout=args.force_patch_dropout,
            force_image_size=args.force_image_size,
            image_mean=args.image_mean,
            image_std=args.image_std,
            image_interpolation=args.image_interpolation,
            image_resize_mode=args.image_resize_mode,  # only effective for inference
            aug_cfg=args.aug_cfg,
            pretrained_image=args.pretrained_image,
            output_dict=True,
            video=args.video, # TODO: add video support in parser
            **model_kwargs,
        )

    # Wrap model with NegationAwareCLIPWrapper if hypothesis testing method is specified
    from src.evaluation.modified_clip import NegationAwareCLIPWrapper
    negation_method = getattr(args, "negation_method", "baseline")
    if negation_method != "baseline":
        print(f"Wrapping model with NegationAwareCLIPWrapper (Method: {negation_method})")
        model = NegationAwareCLIPWrapper(
            model,
            negation_method=negation_method,
            subspace_basis_path=getattr(args, "subspace_basis_path", None),
            hyperplane_weight_path=getattr(args, "hyperplane_weight_path", None),
            hyperplane_lambda=getattr(args, "hyperplane_lambda", 0.5),
            bilinear_alpha=getattr(args, "bilinear_alpha", 0.5)
        )

    random_seed(args.seed, args.rank)
    if is_master(args):
        logging.info("Model:")
        logging.info(f"{str(model)}")
        logging.info("Params:")
        params_file = os.path.join(args.logs, args.name, "params.txt")
        with open(params_file, "w") as f:
            for name in sorted(vars(args)):
                val = getattr(args, name)
                logging.info(f"  {name}: {val}")
                f.write(f"{name}: {val}\n")

    # initialize datasets
    start_epoch = 0
    if args.cxr_dataset and "biomedclip" in args.name:
        tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
    elif "quiltnet" in args.name:
        tokenizer = get_tokenizer('hf-hub:wisdomik/QuiltNet-B-32')
    elif "conch" in args.name:
        from conch.open_clip_custom import get_tokenizer as get_tokenizer_conch
        tokenizer = get_tokenizer_conch()
    else:
        tokenizer = get_tokenizer(args.model)
    data = get_data(
        args,
        (preprocess_train, preprocess_val),
        epoch=start_epoch,
        tokenizer=tokenizer,
    )
    assert len(data), 'At least one train or eval dataset must be specified.'
    print("data keys:")
    for key in data.keys():
        print(key)

    scheduler = None # No training, no scheduler needed

    # determine if this worker should save logs and checkpoints. only do so if it is rank == 0
    args.save_logs = args.logs and args.logs.lower() != 'none' and is_master(args)
    writer = None
    if args.save_logs and args.tensorboard:
        assert tensorboard is not None, "Please install tensorboard."
        writer = tensorboard.SummaryWriter(args.tensorboard_path)

    if args.wandb and is_master(args):
        assert wandb is not None, 'Please install wandb.'
        logging.debug('Starting wandb.')
        # args.train_sz = data["train"].dataloader.num_samples
        if args.val_data is not None:
            args.val_sz = data["val"].dataloader.num_samples
        # you will have to configure this for your project!
        wandb.init(
            project=args.wandb_project_name,
            name=args.name,
            # id=args.name,
            notes=args.wandb_notes,
            tags=[],
            resume='auto' if args.resume == "latest" else None,
            config=vars(args),
        )
        if args.debug:
            wandb.watch(model, log='all')
        wandb.save(params_file)
        logging.debug('Finished loading wandb.')

    # Pytorch 2.0 adds '_orig_mod.' prefix to keys of state_dict() of compiled models.
    # For compatibility, we save state_dict() of the original model, which shares the
    # weights without the prefix.
    original_model = model
    if args.torchcompile:
        logging.info('Compiling model...')
        model = torch.compile(original_model)

    if 'train' not in data:
        # If using int8, convert to inference mode.
        if args.use_bnb_linear is not None:
            from open_clip.utils import convert_int8_model_to_inference_mode
            convert_int8_model_to_inference_mode(model)
        # Evaluate.
        if args.video:
            evaluate_video(model, data, start_epoch, args, tb_writer=writer, tokenizer=tokenizer)
        else:
            evaluate(model, data, start_epoch, args, tb_writer=writer, tokenizer=tokenizer)
        return


if __name__ == "__main__":
    main(sys.argv[1:])