"""Inference and scoring."""

from rris.inference.baseline import predict_baseline, predict_baseline_with_probs
from rris.inference.common import get_hex_color
from rris.inference.embedding import predict_embedding, predict_embedding_with_probs
from rris.inference.prep import prepare_scoring_dataframe
from rris.inference.xlmr import predict_xlmr, predict_xlmr_with_probs
