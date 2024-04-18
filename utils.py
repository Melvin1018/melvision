import torch
import os

def load_pretrained_weights(url):
    file_name = url.split('/')[-1]
    weights_dir = './weights'
    weight_path = os.path.join(weights_dir, file_name)

    if os.path.exists(weight_path):
        state_dict = torch.load(weight_path)
    else:
        if not os.path.exists(weights_dir):
            os.makedirs(weights_dir)

        state_dict = torch.hub.load_state_dict_from_url(url, progress=True)
        torch.save(state_dict, weight_path)

    return file_name, state_dict