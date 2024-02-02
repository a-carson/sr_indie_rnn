import torch
import json
from sr_indie_rnn.modules import RNN
import string


def RNN_from_state_dict(filename: str):
    with open(filename, 'r') as f:
        json_data = json.load(f)

    model_data = json_data["model_data"]

    model = RNN(cell_type=model_data["unit_type"],
                in_channels=model_data["input_size"],
                out_channels=model_data["output_size"],
                hidden_size=model_data["hidden_size"],
                residual_connection=bool(model_data["skip"]))

    state_dict = {}
    for key, value in json_data["state_dict"].items():
        state_dict[key.replace("lin", "linear")] = torch.tensor(value)

    model.load_state_dict(state_dict)
    return model



