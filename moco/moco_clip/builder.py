# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
import torch.nn as nn


class MoCoCLIP(nn.Module):
    def __init__(
        self,
        vision_encoder,
        sentence_encoder="distilbert-base-uncased",
        K=65536,            # buffer size
        T=0.07,             # temperature (fixed)
        mlp=False,          # MLP projection head on ViT-CLS
        load_path=None,
        *args, **kwargs,
    ):
        super(MoCoCLIP, self).__init__()

        self.K = K
        self.T = T

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_k, self.tokenizer_k, self.dim_k = sentence_embedding_manager(sentence_encoder)
        # freeze the text encoder
        for param_k in self.encoder_k.parameters():
            param_k.requires_grad = False

        if load_path not in [None, ""]:
            self.encoder_q = load_pretrained_model(vision_encoder, load_path, self.dim_k)
        else:
            self.encoder_q = vision_encoder(num_classes=self.dim_k)

        if mlp:  # hack: brute-force replacement
            raise NotImplementedError
            # dim_mlp = self.encoder_q.fc.weight.shape[1]
            # self.encoder_q.fc = nn.Sequential(
            #     nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc
            # )

        # create the queue
        self.register_buffer("queue", torch.randn(self.dim_k, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr : ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    def forward(self, im_q, txt_k):
        """
        Input:
            im_q:  a batch of query images
            txt_k: a batch of key text
        Output:
            logits, targets
        """

        # compute query features
        q = self.encoder_q(im_q)                # queries: NxC
        q = nn.functional.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            tokens_k = self.tokenizer_k(txt_k, return_tensors="pt", padding=True, truncation=True)
            tokens_k = {k: v.to(q.device) for k, v in tokens_k.items()}
            # average pool features over the last hidden layer (across token dim)
            k = self.encoder_k(**tokens_k).last_hidden_state.mean(dim=1)  # NxC
            k = nn.functional.normalize(k, dim=1)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum("nc,ck->nk", [q, self.queue.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)

        return logits, labels


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [
        torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


# model loading function
def load_pretrained_model(
    base_encoder: callable,
    load_path: str,
    dim: int,
) -> nn.Module:
    """Load a pretrained model for downstream adaptation with MoCo"""
    # num_classes is the output fc dimension
    model = base_encoder(num_classes=dim)
    checkpoint = torch.load(load_path, map_location=torch.device("cpu"))
    # rename moco pre-trained keys
    state_dict = checkpoint["state_dict"]
    for k in list(state_dict.keys()):
        # retain only encoder_q up to before the embedding layer
        if k.startswith("module.encoder_q") and not k.startswith("module.encoder_q.fc"):
            # remove prefix
            state_dict[k[len("module.encoder_q.") :]] = state_dict[k]
        # delete renamed or unused k
        del state_dict[k]
    msg = model.load_state_dict(state_dict, strict=False)
    assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}
    return model


# LLM loading function
def sentence_embedding_manager(
    model_name: str = "distilbert-base-uncased",
):
    if model_name == "distilbert-base-uncased":
        return load_distilbert_model(cased=False)
    elif model_name == "distilbert-base-cased":
        return load_distilbert_model(cased=True)
    else:
        raise NotImplementedError
    

def load_distilbert_model(cased=False):
    from transformers import AutoTokenizer, AutoModel, AutoConfig
    if cased:
        model_name = "distilbert-base-cased"
    else:
        model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    return model, tokenizer, 768
    

