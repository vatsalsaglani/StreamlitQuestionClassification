from tokenizers import BertWordPieceTokenizer
from TwoClassHeadClassificationTransformer import *
# from ClassificationDatasetFromDict import *
import pickle 
import torch 
import torch.nn as nn 
import numpy as np 
import streamlit as st

SEED = 3007
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)


def save_pickle(obj, filepath):
    with open(filepath, 'wb') as fp:
        pickle.dump(obj, fp)

def load_pickle(filepath):
    with open(filepath, 'rb') as fp:
        return pickle.load(fp)      

@st.cache
def load_model():
    tokenizer = BertWordPieceTokenizer('bert-word-piece-custom-wikitext-vocab-10k-vocab.txt', lowercase = True, strip_accents = True)
    vocab_size = tokenizer.get_vocab_size()
    pad_id = 0
    CLS_label_id = 2
    num_class_heads = 2
    lst_num_cat_in_classes = [6, 47]
    seq_len = 100
    batch_size = 256
    num_workers = 3

    model = TwoClassHeadClassificationTransformer(
        vocab_size=vocab_size, pad_id=pad_id, CLS_label_id=CLS_label_id,
        num_class_heads=num_class_heads, 
        lst_num_cat_in_classes=lst_num_cat_in_classes, num_pos=seq_len
    )
    model = torch.load('classification_model_best.pt', map_location = 'cpu')
    model = model.to('cpu')
    model = model.eval()

    return model

@st.cache
def inf(text, model):
    class2names = {
    "DESC": "DESCRIPTION",
    "ENTY": "ENTITY",
    "ABBR": "ABBREVIATION",
    "HUM": "HUMAN",
    "NUM": "NUMERIC",
    "LOC": "LOCATION"
    }

    class2names = load_pickle('class2names.pkl')
    subclass2names = load_pickle('subclass2names.pkl')
    idx2class = load_pickle('idx2class.pkl')
    idx2subclass = load_pickle('idx2subclass.pkl')

    tokenizer = BertWordPieceTokenizer('bert-word-piece-custom-wikitext-vocab-10k-vocab.txt', lowercase = True, strip_accents = True)

    tokens = torch.FloatTensor(tokenizer.encode(text).ids).unsqueeze(0).to('cpu')
    cls_, subcls = model(tokens)
    clsIdx = cls_.max(1)[-1].item()
    subclsIdx = subcls.max(1)[-1].item()

    return {
        "class": class2names[idx2class[clsIdx]],
        "subclass": subclass2names[idx2subclass[subclsIdx]]
    }
