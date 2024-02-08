import torch
import os
import shutil

from .dataloader_transformer_decoder import DataLoaderTransformerDecoder
from definitions import DATA_DIR, TMP_DIR

# https://huggingface.co/allenai/OLMo-7B
import hf_olmo # pip install ai2-olmo
from transformers import AutoModelForCausalLM

class DataLoaderOLMo(DataLoaderTransformerDecoder):
    """

    Notes:
    ------
    OLMo-1B has the following architecture: 

        OLMoForCausalLM(
        (model): Olmo(
            (transformer): ModuleDict(
            (wte): Embedding(50304, 2048)
            (emb_drop): Dropout(p=0.0, inplace=False)
            (ln_f): LayerNorm()
            (blocks): ModuleList(
                (0-15): 16 x OlmoSequentialBlock(
                (dropout): Dropout(p=0.0, inplace=False)
                (act): SwiGLU()
                (attn_out): Linear(in_features=2048, out_features=2048, bias=False)
                (ff_out): Linear(in_features=8192, out_features=2048, bias=False)
                (rotary_emb): RotaryEmbedding()
                (attn_norm): LayerNorm()
                (ff_norm): LayerNorm()
                (att_proj): Linear(in_features=2048, out_features=6144, bias=False)
                (ff_proj): Linear(in_features=2048, out_features=16384, bias=False)
                )
            )
            (ff_out): Embedding(50304, 2048)
            )
        )
        )
    """
    def __init__(self, model_name, steps_list):

        model_args = {}
        if model_name == "OLMo-1B":
            model_args['n_embd'] = 2048
            model_args['n_layer'] = 16
            model_args['n_head'] = 16
        elif model_name == "OLMo-7B":
            model_args['n_embd'] = 4096
            model_args['n_layer'] = 32
            model_args['n_head'] = 32
        elif model_name == "OLMo-7B-Twin-2T":
            model_args['n_embd'] = 4096
            model_args['n_layer'] = 32
            model_args['n_head'] = 32

        super(DataLoaderOLMo, self).__init__(
            model_name,
            model_args,
            steps_list
            )

        self.cache_dir = os.path.join(TMP_DIR, 'olmo-cache')

    def set_path(self):
        self.path = os.path.join(DATA_DIR, 'OLMo')

    def get_revision_name(self, step):
        fname = os.path.join(DATA_DIR, 'OLMo', self.model_name + '-revisions.txt')
        with open(fname) as f:
            for line in f.readlines():
                line = line.strip()
                if line.startswith('step'+str(step)+'-tokens'):
                    return line
        return None

    def get_ckpt(self, step):
        assert step in self.steps_list
        revision = self.get_revision_name(step)
        if revision is None:
            print('Attempted to retrieve the revision for step', step)
            raise ValueError('there is no such revision name for model', self.model_name)

        model = AutoModelForCausalLM.from_pretrained(
            "allenai/" + self.model_name,
            revision=revision,
            cache_dir=self.cache_dir
        )
        return model

    def get_device(self, model):
        return model.model.transformer.wte.weight.device

    def get_attn_QKV_weight(self, model, i_layer):
        return model.model.transformer.blocks[i_layer].att_proj.weight.data

    def get_attn_O_weight(self, model, i_layer):
        return model.model.transformer.blocks[i_layer].attn_out.weight.data.T

    def get_emb_in(self, model):
        return model.model.transformer.wte.weight.T

    def get_emb_out(self, model):
        return None

    def get_mlp_in(self, model, i_layer):
        return model.model.transformer.blocks[i_layer].ff_proj.weight.T 

    def get_mlp_out(self, model, i_layer):
        return model.model.transformer.blocks[i_layer].ff_out.weight 

    def get_ln_1(self, model, i_layer):
        return model.model.transformer.blocks[i_layer].attn_norm.weight

    def get_ln_2(self, model, i_layer):
        return model.model.transformer.blocks[i_layer].ff_norm.weight

    def process(self):
        # an empty implementation of this abstract class
        pass