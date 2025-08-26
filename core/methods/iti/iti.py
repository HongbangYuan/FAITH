import torch
from baukit import Trace, TraceDict
from utils.nethook import TraceDict,Trace

def get_llama_activations_bau(model, prompt):

    model.eval()

    HEADS = [f"model.layers.{i}.self_attn.o_proj" for i in range(model.config.num_hidden_layers)]
    # MLPS = [f"model.layers.{i}.mlp" for i in range(model.config.num_hidden_layers)]
    MLPS = []

    with torch.no_grad():
        prompt = prompt.to(model.device)
        with TraceDict(model, HEADS+MLPS, retain_output=True, retain_input=True, clone=True, detach=True) as ret:
            output = model(prompt, output_hidden_states = True)
        hidden_states = output.hidden_states
        hidden_states = torch.stack(hidden_states, dim = 0).squeeze()
        hidden_states = hidden_states.detach().cpu().numpy()
        head_wise_hidden_states = [ret[head].input.squeeze().detach().cpu() for head in HEADS]
        head_wise_hidden_states = torch.stack(head_wise_hidden_states, dim = 0).squeeze().numpy()
        # mlp_wise_hidden_states = [ret[mlp].output.squeeze().detach().cpu() for mlp in MLPS]
        # mlp_wise_hidden_states = torch.stack(mlp_wise_hidden_states, dim = 0).squeeze().numpy()

    return hidden_states, head_wise_hidden_states

