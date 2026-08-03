import logging

import torch
import torch.nn.functional as F
from tqdm import tqdm

from open_clip import get_input_dtype, get_tokenizer
from training.precision import get_autocast


def evaluate_model(model, dataloader, args, tokenizer, recall_k_list=[5]):
    """
    Evaluates a model on retrieval tasks.
    Args:
        model: torch.nn.Module
            The model to evaluate.
        dataloader: torch.utils.data.DataLoader
            The dataloader to evaluate on.
        args: argparse.Namespace
            The command line arguments.
        tokenizer: transformers.PreTrainedTokenizer
            The tokenizer to use.
        recall_k_list: list
            A list of k values for recall@k.
    Returns:
        metrics: dict
            The evaluation metrics.
    """
    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)
    if tokenizer is None:
        tokenizer = get_tokenizer(args.model)
    model.eval()

    batch_images_emb_list = []
    batch_texts_emb_list = []
    texts_image_index = []
    dataloader = dataloader_with_indices(dataloader)

    # Iterate over the dataloader
    for batch_images, batch_texts, inds in tqdm(dataloader):
        batch_images = batch_images.to(device=args.device, dtype=input_dtype)
        batch_texts_tok = tokenizer([text for i, texts in enumerate(batch_texts) for text in texts]).to(args.device)
        batch_texts_image_index = [ind for ind, texts in zip(inds, batch_texts) for text in texts]

        # Compute the embeddings
        with torch.no_grad(), autocast():
            batch_images_emb = F.normalize(model.encode_image(batch_images), dim=-1)
            batch_texts_emb = F.normalize(model.encode_text(batch_texts_tok), dim=-1)
        
        # Append the embeddings and indices
        batch_images_emb_list.append(batch_images_emb.cpu())
        batch_texts_emb_list.append(batch_texts_emb.cpu())
        texts_image_index.extend(batch_texts_image_index)

    batch_size = len(batch_images_emb_list[0])
    images_emb = torch.cat(batch_images_emb_list)
    texts_emb = torch.cat(batch_texts_emb_list)

    # Compute the scores
    if hasattr(model, "compute_retrieval_similarity"):
        scores = model.compute_retrieval_similarity(texts_emb, images_emb)
    else:
        scores = texts_emb @ images_emb.t()
    positive_pairs = torch.zeros_like(scores, dtype=bool)
    positive_pairs[torch.arange(len(scores)), texts_image_index] = True
    metrics = {}

    # Compute the recall@k
    for recall_k in recall_k_list:
        metrics[f"image_retrieval_recall@{recall_k}"] = (batchify(recall_at_k, scores, positive_pairs, batch_size, args.device, k=recall_k)>0).float().mean().item()
        metrics[f"text_retrieval_recall@{recall_k}"] = (batchify(recall_at_k, scores.T, positive_pairs.T, batch_size, args.device, k=recall_k)>0).float().mean().item()
    return metrics

def retrieval_eval(model, data, args, tokenizer=None):
    """
    Evaluates a model on retrieval tasks.
    Args:
        model: torch.nn.Module
            The model to evaluate.
        data: dict
            A dictionary of datasets.
        args: argparse.Namespace
            The command line arguments.
        tokenizer: transformers.PreTrainedTokenizer
            The tokenizer to use.
    Returns:
        results: dict
            The evaluation results.
    """
    results = {}

    if args.video:
        if 'msrvtt-retrieval' in data:
            logging.info('Evaluating on the MSR-VTT retrieval task')
            msrvtt_retrieval = evaluate_model(model, data['msrvtt-retrieval'].dataloader, args, tokenizer=tokenizer, recall_k_list=[1, 5])
            results['msrvtt-video_retrieval_recall@1'] = msrvtt_retrieval['image_retrieval_recall@1']
            results['msrvtt-video_retrieval_recall@5'] = msrvtt_retrieval['image_retrieval_recall@5']
        
        if 'msrvtt-negated-retrieval' in data:
            logging.info('Evaluating on the MSR-VTT negated retrieval task')
            msrvtt_negated_retrieval = evaluate_model(model, data['msrvtt-negated-retrieval'].dataloader, args, tokenizer=tokenizer, recall_k_list=[1, 5])
            results['msrvtt-negated-video_retrieval_recall@1'] = msrvtt_negated_retrieval['image_retrieval_recall@1']
            results['msrvtt-negated-video_retrieval_recall@5'] = msrvtt_negated_retrieval['image_retrieval_recall@5']

    else:
        if 'coco-retrieval' in data:
            logging.info('Evaluating on the COCO retrieval task')
            coco_retrieval = evaluate_model(model, data['coco-retrieval'].dataloader, args, tokenizer=tokenizer, recall_k_list=[1, 5])
            results['coco-image_retrieval_recall@1'] = coco_retrieval['image_retrieval_recall@1']
            results['coco-image_retrieval_recall@5'] = coco_retrieval['image_retrieval_recall@5']

        if 'coco-negated-retrieval' in data:
            logging.info('Evaluating on the COCO negated retrieval task')
            coco_negated_retrieval = evaluate_model(model, data['coco-negated-retrieval'].dataloader, args, tokenizer=tokenizer, recall_k_list=[1, 5])
            results['coco-negated-image_retrieval_recall@1'] = coco_negated_retrieval['image_retrieval_recall@1']
            results['coco-negated-image_retrieval_recall@5'] = coco_negated_retrieval['image_retrieval_recall@5']

    return results

def dataloader_with_indices(dataloader):
    """
    Yields batches of data with indices.
    Args:
        dataloader: torch.utils.data.DataLoader
            The dataloader to iterate over.
    Yields:
        x: torch.Tensor
            The input data.
        y: torch.Tensor
            The target data.
        inds: torch.Tensor
            The indices matching y to x.
    """
    start = 0
    for x, y in dataloader:
        end = start + len(x)
        inds = torch.arange(start, end)
        yield x, y, inds
        start = end

def recall_at_k(scores, positive_pairs, k):
    """
    Computes recall@k for a given set of scores and positive pairs.
    Args:
        scores: torch.Tensor
            The scores of the model.
        positive_pairs: torch.Tensor
            A binary tensor indicating positive pairs.
        k: int
            The value of k for recall@k.
    Returns:
        recall_at_k: torch.Tensor
            The recall@k value.
    """
    nb_texts, nb_images = scores.shape
    topk_indices = torch.topk(scores, k, dim=1)[1]
    nb_positive = positive_pairs.sum(dim=1)
    topk_indices_onehot = torch.nn.functional.one_hot(topk_indices, num_classes=nb_images)
    positive_pairs_reshaped = positive_pairs.view(nb_texts, 1, nb_images)
    nb_true_positive = (topk_indices_onehot * positive_pairs_reshaped).sum(dim=(1,2))
    recall_at_k = (nb_true_positive / nb_positive)
    return recall_at_k


def batchify(func, X, Y, batch_size, device, *args, **kwargs):
    """
    Applies a function to batches of data.
    Args:
        func: callable
            The function to apply.
        X: torch.Tensor
            The input data.
        Y: torch.Tensor
            The target data.
        batch_size: int
            The batch size.
        device: torch.device
            The device to use.
        *args: list
            Additional positional arguments to pass to func.
        **kwargs: dict
            Additional keyword arguments to pass to func.
    Returns:
        results: torch.Tensor
            The results of applying func to the data.
    """
    results = []
    for start in range(0, len(X), batch_size):
        end = start + batch_size
        x = X[start:end].to(device)
        y = Y[start:end].to(device)
        result = func(x, y, *args, **kwargs).cpu()
        results.append(result)
    return torch.cat(results)