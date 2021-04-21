import os
import numpy as np
import torch
import torch.nn as nn
import transformers
from transformers.models.gpt2.modeling_gpt2 import GPT2Model, GPT2LMHeadModel
from torch.nn.parameter import Parameter

def load_tf_weights_in_image_gpt2(model, config, gpt2_checkpoint_path):
    """ Load tf checkpoints in a pytorch model
    """
    try:
        import re
        import tensorflow as tf
    except ImportError:
        print(
            "Loading a TensorFlow model in PyTorch, requires TensorFlow to be installed. Please see "
            "https://www.tensorflow.org/install/ for installation instructions."
        )
        raise
    tf_path = os.path.abspath(gpt2_checkpoint_path)
    print("Converting TensorFlow checkpoint from {}".format(tf_path))
    # Load weights from TF model
    init_vars = tf.train.list_variables(tf_path)
    names = []
    arrays = []

    for name, shape in init_vars:
        print("Loading TF weight {} with shape {}".format(name, shape))
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        arrays.append(array.squeeze())

    for name, array in zip(names, arrays):
        name = name[6:]  # skip "model/"
        name = name.split("/")

        # adam_v and adam_m are variables used in AdamWeightDecayOptimizer to calculated m and v
        # which are not required for using pretrained model
        if any(
            n in ["adam_v", "adam_m", "AdamWeightDecayOptimizer", "AdamWeightDecayOptimizer_1", "global_step"]
            for n in name
        ) or name[-1] in ['_step']:
            print("Skipping {}".format("/".join(name)))
            continue
        
        pointer = model
        if name[-1] not in ["wtet"]:
          pointer = getattr(pointer, "transformer")
        
        for m_name in name:
            if re.fullmatch(r"[A-Za-z]+\d+", m_name):
                scope_names = re.split(r"(\d+)", m_name)
            else:
                scope_names = [m_name]

            if scope_names[0] == "w" or scope_names[0] == "g":
                pointer = getattr(pointer, "weight")
            elif scope_names[0] == "b":
                pointer = getattr(pointer, "bias")
            elif scope_names[0] == "wpe" or scope_names[0] == "wte":
                pointer = getattr(pointer, scope_names[0])
                pointer = getattr(pointer, "weight")
            elif scope_names[0] in ['q_proj','k_proj','v_proj']:
                pointer = getattr(pointer, 'c_attn')
                pointer = getattr(pointer, 'weight')
            elif len(name) ==3 and name[1]=="attn" and scope_names[0]=="c_proj":
                pointer = getattr(pointer, scope_names[0])
                pointer = getattr(pointer, 'weight')
            elif scope_names[0]=="wtet":
                pointer = getattr(pointer, "lm_head")
                pointer = getattr(pointer, 'weight')
            elif scope_names[0]=="sos":
                pointer = getattr(pointer,"wte")
                pointer = getattr(pointer, 'weight')
            else:
                pointer = getattr(pointer, scope_names[0])
            if len(scope_names) >= 2:
                num = int(scope_names[1])
                pointer = pointer[num]

        if len(name) > 1 and name[1]=="attn" or name[-1]=="wtet" or name[-1]=="sos" or name[-1]=="wte":
           pass  # array is used to initialize only part of the pointer so sizes won't match
        else:
          try:
              assert pointer.shape == array.shape
          except AssertionError as e:
              e.args += (pointer.shape, array.shape)
              raise
          
        print("Initialize PyTorch weight {}".format(name))

        if name[-1]=="q_proj":
          pointer.data[:,:config.n_embd] = torch.from_numpy(array.reshape(config.n_embd, config.n_embd)).T
        elif name[-1]=="k_proj":
          pointer.data[:,config.n_embd:2*config.n_embd] = torch.from_numpy(array.reshape(config.n_embd, config.n_embd)).T
        elif name[-1]=="v_proj":
          pointer.data[:,2*config.n_embd:] = torch.from_numpy(array.reshape(config.n_embd, config.n_embd)).T
        elif (len(name) ==3 and name[1]=="attn" and name[2]=="c_proj" ):
          pointer.data = torch.from_numpy(array.reshape(config.n_embd, config.n_embd))
        elif name[-1]=="wtet":
          pointer.data = torch.from_numpy(array)
        elif name[-1]=="wte":
          pointer.data[:config.vocab_size-1,:] = torch.from_numpy(array)
        elif name[-1]=="sos":
          pointer.data[-1] = torch.from_numpy(array)
        else:
          pointer.data = torch.from_numpy(array)

    return model

class ln_mod(nn.Module):
    def __init__(self, nx, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = Parameter(torch.Tensor(nx))

    def forward(self, x):  # input is not mean centered
        return x / torch.sqrt(torch.std(x, axis=-1, unbiased=False, keepdim=True)**2 + self.eps ) * self.weight.data[...,:] 

def replace_ln(m, name, config):
  for attr_str in dir(m):
      target_attr = getattr(m, attr_str)
      if type(target_attr) == torch.nn.LayerNorm:
          #print('replaced: ', name, attr_str)
          setattr(m, attr_str, ln_mod(config.n_embd, config.layer_norm_epsilon))

  for n, ch in m.named_children():
      replace_ln(ch, n, config)        

def gelu2(x):
    return x * torch.sigmoid(1.702 * x)

class ImageGPT2LMHeadModel(GPT2LMHeadModel):
  load_tf_weights = load_tf_weights_in_image_gpt2
  
  def __init__(self, config):
      super().__init__(config)
      self.lm_head = nn.Linear(config.n_embd, config.vocab_size - 1, bias=False)
      replace_ln(self, "net", config)  # replace layer normalization
      for n in range(config.n_layer):
        self.transformer.h[n].mlp.act = gelu2  # replace activation 

  def tie_weights(self):  # image-gpt doesn't tie output and input embeddings
    pass 

class LinearProbeImageGPT(nn.Module):
    """ Image GPT with a linear classifier head attached """
    def __init__(self, wte, wpe, drop, blocks, ln_1, head):
        super().__init__()

        # input embedding stem
        self.tok_emb = wte
        self.pos_emb = wpe
        self.drop = drop
        self.blocks = blocks
        self.ln_1 = ln_1
        self.head = head

        print('Number of parameters in LinearProbeImageGPT:', sum(p.numel() for p in self.parameters()))

    def forward(self, idx):
        _, t = idx.size()

        pos_idx = torch.arange(0, t, dtype=torch.long)
        pos_idx = pos_idx.cuda()
        pos_idx = pos_idx.unsqueeze(0).view(-1, t)

        # forward the GPT model
        token_embeddings = self.tok_emb(idx)  
        position_embeddings = self.pos_emb(pos_idx) 

        x = self.drop(token_embeddings + position_embeddings)
        for block in range(len(self.blocks)):
            x = self.blocks[block](x)[0]

        x = self.ln_1(x)
        x = torch.mean(x, 1, False)
        logits = self.head(x)

        return logits        

def load_igpt(model_size, model_path, cluster_path, n_px, prly, n_classes):
    """ Load pretrained model and clusters """
    if model_size == "l":
        n_embd, n_head, n_layer = 1536, 16, 48
    elif model_size == "m":
        n_embd, n_head, n_layer = 1024, 8, 36
    elif model_size == "s":
        n_embd, n_head, n_layer = 512, 8, 24

    clusters = np.load(cluster_path)  # get color clusters

    vocab_size = len(clusters) + 1  # add one for start of sentence token
    config = transformers.GPT2Config(vocab_size=vocab_size, n_ctx=n_px*n_px, n_positions=n_px*n_px, n_embd=n_embd, n_layer=n_layer, n_head=n_head)
    model = ImageGPT2LMHeadModel.from_pretrained(model_path, from_tf=True, config=config)
    print(model.transformer.h[:prly])

    head = torch.nn.Linear(in_features=n_embd, out_features=n_classes, bias=True)
    model = LinearProbeImageGPT(model.transformer.wte, model.transformer.wpe, model.transformer.drop, model.transformer.h[:prly], model.transformer.h[prly+1].ln_1, head)

    return model, torch.from_numpy(clusters)