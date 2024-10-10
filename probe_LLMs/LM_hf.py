from transformers import LlavaForConditionalGeneration,AutoModelForCausalLM, AutoTokenizer
from nnsight import LanguageModel
import numpy as np
from transformers.generation.utils import GenerationConfig

from llava.mm_utils import (
    KeywordsStoppingCriteria,
    expand2square,
    get_model_name_from_path,
    load_pretrained_model,
    tokenizer_image_token,
    tokenizer_image_token_with_mask,
)
from llava.model.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX, key_info
from llava.conversation import Conversation, SeparatorStyle
import os
import torch

class LM_nnsight():
    def __init__(self, model_path, device="cuda", temperature=0.,cahce_dir=None):
        self.device = device
        self.model_path = model_path

        if "Yi" in model_path and "VL" in model_path:
            model_path_expand = os.path.expanduser(model_path)
            key_info["model_path"] = model_path_expand
            tokenizer, base_model, _, _ = load_pretrained_model(model_path)
        else:
            if "llava" in model_path:
                base_model = LlavaForConditionalGeneration.from_pretrained(model_path, trust_remote_code=True, cache_dir=cahce_dir,device_map="auto")
            else:
                base_model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, cache_dir=cahce_dir,device_map="auto")
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, cache_dir=cahce_dir,device_map="auto")
        if tokenizer.pad_token is None:
            #tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            if "Qwen" in model_path and tokenizer.eos_token is None:
                tokenizer.eos_token='<|endoftext|>'
            tokenizer.pad_token = tokenizer.eos_token
            base_model.resize_token_embeddings(len(tokenizer))

        base_model.generation_config = GenerationConfig.from_pretrained(model_path,trust_remote_code=True)
        if temperature == 0:
            base_model.generation_config.do_sample = False
            base_model.generation_config.temperature = None
            base_model.generation_config.top_p = None
            base_model.generation_config.top_k = None
            #print(base_model.generation_config.temperature)
        else:
            base_model.generation_config.temperature = temperature
        #base_model.to(self.device)
        base_model.eval()

        if model_path == "Qwen/Qwen-7B-Chat":
            self.model = LanguageModel("Qwen/Qwen-7B-Chat", tokenizer=tokenizer,device_map="auto",dispatch=True,cache_dir=cahce_dir,trust_remote_code=True)
        elif "MiniCPM" in model_path:
            self.model = LanguageModel(base_model.llm, tokenizer=tokenizer,device_map="auto",dispatch=True,cache_dir=cahce_dir,trust_remote_code=True)
        else:
            self.model = LanguageModel(base_model, tokenizer=tokenizer,device_map="auto",cache_dir=cahce_dir,trust_remote_code=True)
        
    def generate_response(self, prompt, max_new_tokens=2000):
        with self.model.generate(max_new_tokens=max_new_tokens) as generator:  
            with generator.invoke(prompt) as invoker:
                pass
        return self.model.tokenizer.decode(generator.output[0])
                
    def __call__(self, prompt, max_new_tokens=2000):
        ans = self.generate_response(prompt, max_new_tokens)
        return ans

    def get_all_states(self, prompt):

        if "llava" in self.model_path:
            n_heads = self.model.language_model.config.num_attention_heads
        elif "Qwen" in self.model_path and "1.5" not in self.model_path:
            n_heads = self.model.transformer.h[0].attn.num_heads 
        elif "MiniCPM" in self.model_path:
            n_heads = self.model.model.config.num_attention_heads
        else:
            n_heads = self.model.model.config.num_attention_heads
        
        all_hidden_states = []
        all_attention_states = []
        
        #with self.model.invoke(prompt) as invoker:

        if "Qwen" in self.model_path and "1.5" not in self.model_path:
            with self.model.trace(prompt,scan=False,validate=False):
                for layer in self.model.transformer.h:
                    #layer.self_attn.output[0] shape [batch_size, seq_len, hidden_size]
                    all_attention_states.append(layer.attn.output[0].save())
                    all_hidden_states.append(layer.output[0].save()) 
                    #hidden_states is indexed by output[0]:
        elif "cogvlm2" in self.model_path:
            with self.model.trace(prompt,scan=False,validate=False):
                for layer in self.model.model.layers:
                    #layer.self_attn.output[0] shape [batch_size, seq_len, hidden_size]
                    # print(layer.self_attn.output[0].shape)
                    # print(layer.self_attn.output[0].save())
                    # assert False
                    all_attention_states.append(layer.self_attn.output[0].save()) 
                    all_hidden_states.append(layer.output[0].save())
        elif "MiniCPM" in self.model_path:
            with self.model.trace(prompt,scan=False,validate=False):
                for layer in self.model.model.layers:
                    #layer.self_attn.output[0] shape [batch_size, seq_len, hidden_size]
                    all_attention_states.append(layer.self_attn.output[0].save()) 
                    all_hidden_states.append(layer.output[0].save())
                    #hidden_states is indexed by output[0]:
                    #all_hidden_states.append(layer.self_attn.output[0].save()) 


        else:
            with self.model.trace(prompt):
                if "llava" in self.model_path:
                    for layer in self.model.language_model.model.layers:
                        #layer.self_attn.output[0] shape [batch_size, seq_len, hidden_size]
                        all_attention_states.append(layer.self_attn.output[0].save()) 
                        #hidden_states is indexed by output[0]: https://github.com/ndif-team/nnsight
                        all_hidden_states.append(layer.output[0].save())
                else:
                    for layer in self.model.model.layers:
                        #layer.self_attn.output[0] shape [batch_size, seq_len, hidden_size]
                        all_attention_states.append(layer.self_attn.output[0].save()) 
                        #hidden_states is indexed by output[0]: https://github.com/ndif-team/nnsight
                        all_hidden_states.append(layer.output[0].save())                
        
        all_hidden_states_numpy = []
        all_attention_states_numpy = []

        for HS, AS in zip(all_hidden_states, all_attention_states):
            #HS: (batch_size, seq_len, hidden_size)
            #AS: (batch_size, seq_len, hidden_size)
            
            #hidden_state_numpy = HS.value[0].cpu().detach().numpy()
            hidden_state = HS.value[0].cpu().detach()
            if hidden_state.dtype == torch.bfloat16:
                hidden_state = hidden_state.to(torch.float32)
            hidden_state_numpy = hidden_state.numpy()

            #attention_state_numpy = AS.value[0].cpu().detach().numpy()
            attention_state = AS.value[0].cpu().detach()
            if attention_state.dtype == torch.bfloat16:
                attention_state = attention_state.to(torch.float32)
            attention_state_numpy = attention_state.numpy()
            
            all_hidden_states_numpy.append(hidden_state_numpy)
            atts = attention_state_numpy
            all_attention_states_numpy.append(atts.reshape(atts.shape[0], n_heads, -1))

        all_hidden_states_numpy = np.array(all_hidden_states_numpy)
        all_attention_states_numpy = np.array(all_attention_states_numpy)

        return all_hidden_states_numpy, all_attention_states_numpy
        # all_hidden_states: (Layers, Tokens, 4096)
        # all_attention_states: (Layers, Tokens, Heads, 128)

    def intervention(self, prompt, interventions_dict, alpha=10, max_new_tokens=3):
        n_layers = len(self.model.model.layers)
        n_heads = self.model.model.config.num_attention_heads
        head_dim = int(self.model.model.config.hidden_size / n_heads)
        with self.model.generate(max_new_tokens=max_new_tokens) as generator:
            with generator.invoke(prompt) as invoker:
                for idx in range(max_new_tokens):
                    for layer_id, layer in enumerate(self.model.model.layers):
                        if layer_id in interventions_dict:
                            for (head, dir, std, _) in interventions_dict[layer_id]:
                                layer.self_attn.output[0][0, -1, head * head_dim: (head + 1) * head_dim] += alpha * std * dir
                    invoker.next()
        return self.model.tokenizer.decode(generator.output[0])








        