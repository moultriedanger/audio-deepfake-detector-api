import torch
import numpy as np
import yaml
from torch.nn import functional as F
from model.model import RawNet
from torch import Tensor
import librosa

def pad(x, max_len=96000):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    num_repeats = int(max_len / x_len)+1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x

def load_sample(sample_path, max_len = 96000):
    y_list = []
    y, sr = librosa.load(sample_path, sr=None)

    if sr != 24000:
        y = librosa.resample(y, orig_sr = sr, target_sr = 24000)

    if len(y) <= 96000:
        return [Tensor(pad(y, max_len))]

    for i in range(int(len(y)/96000)):
        if (i+1) == range(int(len(y)/96000)):
            y_seg = y[i*96000:]
        else:
            y_seg = y[i*96000 : (i+1)*96000]

        y_pad = pad(y_seg, max_len)
        y_inp = Tensor(y_pad)
        y_list.append(y_inp)

    return y_list

def run_inference(input_path, model_path, config_path='model/model_config_RawNet.yaml'):
    with open(config_path, 'r') as f_yaml:
        config = yaml.safe_load(f_yaml)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = RawNet(config['model'], device).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    out_list_multi = []
    out_list_binary = []

    for m_batch in load_sample(input_path):
        m_batch = m_batch.to(device=device, dtype=torch.float).unsqueeze(0)
        logits, multi_logits = model(m_batch)

        probs = F.softmax(logits, dim=-1)
        probs_multi = F.softmax(multi_logits, dim=-1)

        out_list_multi.append(probs_multi.tolist()[0])
        out_list_binary.append(probs.tolist()[0])

    result_multi = np.average(out_list_multi, axis=0).tolist()
    result_binary = np.average(out_list_binary, axis=0).tolist()

    return {
        "binary_classification": {
            "fake": result_binary[0],
            "real": result_binary[1]
        },
        "multi_classification": {
            "gt": result_multi[0],
            "wavegrad": result_multi[1],
            "diffwave": result_multi[2],
            "parallel_wave_gan": result_multi[3],
            "wavernn": result_multi[4],
            "wavenet": result_multi[5],
            "melgan": result_multi[6]
        }
    }
