"""
Flower client: represents one hospital node's local training/eval loop.

Owner: Track B (merges with Track A's data partition + DP engine in sprint 2)

TODO:
- Replace dummy data with a real client-specific shard from
  src/data/partition.py.
- Wrap the local optimizer with src/privacy/dp_engine.py's make_private().
- Only send LoRA adapter weights (get_peft_model_state_dict) in get_parameters
  / set_parameters -- NOT the full frozen base model. This is the whole
  bandwidth-saving point of the architecture.
"""

import flwr as fl
from peft import get_peft_model_state_dict, set_peft_model_state_dict

from src.models.lora_model import build_lora_model


class HospitalClient(fl.client.NumPyClient):
    def __init__(self, client_id: int, r: int = 8):
        self.client_id = client_id
        self.model, self.tokenizer = build_lora_model(r=r)
        # TODO: load this client's data shard here

    def get_parameters(self, config):
        state_dict = get_peft_model_state_dict(self.model)
        return [val.cpu().numpy() for val in state_dict.values()]

    def set_parameters(self, parameters):
        state_dict = get_peft_model_state_dict(self.model)
        keys = list(state_dict.keys())
        new_state_dict = {k: v for k, v in zip(keys, parameters)}
        set_peft_model_state_dict(self.model, new_state_dict)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        # TODO: local training loop (with DP engine wrapped optimizer)
        # TODO: return updated LoRA params, num_examples, metrics
        return self.get_parameters(config={}), 0, {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        # TODO: local eval loop, return loss, num_examples, metrics (incl. F1)
        return 0.0, 0, {"f1": 0.0}


def make_client_fn(r: int = 8):
    def client_fn(cid: str):
        return HospitalClient(client_id=int(cid), r=r).to_client()
    return client_fn
