"""
# -*- coding: utf-8 -*-

"""
import torch


def mean_pooling(model_output, attention_mask):
    """
    均值池化

    :param model_output:
    :param attention_mask:
    :return:
    """
    token_embeddings = model_output  # First element of model_output contains all token embeddings
    # print("token_embeddings",token_embeddings)
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)