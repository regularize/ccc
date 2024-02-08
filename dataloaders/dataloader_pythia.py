import torch
import os
import shutil

from .dataloader_transformer_decoder import DataLoaderTransformerDecoder
from definitions import DATA_DIR, TMP_DIR

from transformers import GPTNeoXForCausalLM

class DataLoaderPythia(DataLoaderTransformerDecoder):
    """

    Notes:
    ------
    "pythia-70m-deduped" has the following architecture:
        GPTNeoXForCausalLM(
        (gpt_neox): GPTNeoXModel(
            (embed_in): Embedding(50304, 512)
            (emb_dropout): Dropout(p=0.0, inplace=False)
            (layers): ModuleList(
            (0-5): 6 x GPTNeoXLayer(
                (input_layernorm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
                (post_attention_layernorm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
                (post_attention_dropout): Dropout(p=0.0, inplace=False)
                (post_mlp_dropout): Dropout(p=0.0, inplace=False)
                (attention): GPTNeoXAttention(
                (rotary_emb): GPTNeoXRotaryEmbedding()
                (query_key_value): Linear(in_features=512, out_features=1536, bias=True)
                (dense): Linear(in_features=512, out_features=512, bias=True)
                (attention_dropout): Dropout(p=0.0, inplace=False)
                )
                (mlp): GPTNeoXMLP(
                (dense_h_to_4h): Linear(in_features=512, out_features=2048, bias=True)
                (dense_4h_to_h): Linear(in_features=2048, out_features=512, bias=True)
                (act): GELUActivation()
                )
            )
            )
            (final_layer_norm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        )
        (embed_out): Linear(in_features=512, out_features=50304, bias=False)
        )
    """
    def __init__(self, model_name, steps_list):

        model_args = {}
        if model_name == "pythia-70m-deduped":
            model_args['n_embd'] = 512
            model_args['n_layer'] = 6
            model_args['n_head'] = 8
        elif model_name == "pythia-160m":
            model_args['n_embd'] = 768
            model_args['n_layer'] = 12
            model_args['n_head'] = 12
        elif model_name == "pythia-410m":
            model_args['n_embd'] = 1024
            model_args['n_layer'] = 24
            model_args['n_head'] = 16
        elif model_name == "pythia-1b":
            model_args['n_embd'] = 2048
            model_args['n_layer'] = 16
            model_args['n_head'] = 8
        elif model_name == "pythia-12b":
            model_args['n_embd'] = 5120
            model_args['n_layer'] = 36
            model_args['n_head'] = 40

        super(DataLoaderPythia, self).__init__(
            model_name,
            model_args,
            steps_list
            )

        self.cache_dir = os.path.join(TMP_DIR, 'pythia-cache')

    def set_path(self):
        self.path = os.path.join(DATA_DIR, 'Pythia')

    def get_revision_name(self, step):
        return "step" + str(step)

    def get_ckpt(self, step):
        assert step in self.steps_list
        model = GPTNeoXForCausalLM.from_pretrained(
            "EleutherAI/" + self.model_name,
            revision=self.get_revision_name(step),
            cache_dir=self.cache_dir)
        return model

    def get_device(self, model):
        return model.gpt_neox.embed_in.weight.device

    def get_attn_QKV_weight(self, model, i_layer):
        return model.gpt_neox.layers[i_layer].attention.query_key_value.weight.data

    def get_attn_O_weight(self, model, i_layer):
        return model.gpt_neox.layers[i_layer].attention.dense.weight.data.T

    def get_emb_in(self, model):
        return model.gpt_neox.embed_in.weight.T

    def get_emb_out(self, model):
        return model.embed_out.weight.T

    def get_mlp_in(self, model, i_layer):
        return model.gpt_neox.layers[i_layer].mlp.dense_h_to_4h.weight.T

    def get_mlp_out(self, model, i_layer):
        return model.gpt_neox.layers[i_layer].mlp.dense_4h_to_h.weight

    def get_ln_1(self, model, i_layer):
        return model.gpt_neox.layers[i_layer].input_layernorm.weight

    def get_ln_2(self, model, i_layer):
        return model.gpt_neox.layers[i_layer].post_attention_layernorm.weight


    def process(self):
        if self.model_name == 'pythia-1b':
            # the 'config' file for step 116000 is missing from Huggingface
            self.remove_steps([116000])
            print('steps_list:', self.steps_list)
        return