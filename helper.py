import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from transformer import Transformer

def val_loss(Dval, model, criterion, vocab_size, mask):
    """
    Compute the validation loss.
    
    Parameters:
    Dval: torch.tensor of shape (context_size, num_batches)
        The validation data already reshaped into the subsequences. It contains the full validation data set. 
    model: nn.Module
        The model to use for the validation
    criterion: callable function
        The loss function to compare the model output to the target. This should be nn.CrossEntropyLoss()
    vocab_size: int
        The number of characters in the vocabulary
    mask: torch.Tensor of shape (context_size, context_size)
        The mask to apply to the attention weights

    Returns:    
    float
        The validation loss
    """
    model.eval()  # set the model to evaluation mode which improves efficiency since no gradients are computed
    with torch.no_grad():
        x = Dval[:-1, :]
        y_pred = model(x, mask)
        y = Dval[1:, :]
        y_pred = y_pred.reshape(-1, vocab_size)
        y = y.reshape(-1)
        vloss = criterion(y_pred, y)
    return vloss.item()

def model_name(esize, cs, nl, nh, epoch, pos_enc=True):
    """
    Create a name for the model based on the parameters.
    Note that this does not differentiate models with different widths in the feedforward network. Also does not include positional encoding in parameters.
    It also does not include the folder name. 

    Parameters:
    esize: int
        The embedding size
    cs: int
        The context size
    nl: int
        The number of transformer layers
    nh: int
        The number of heads in the multiheadattention models
    epoch: int
        The number of epochs the model has been trained for
    pos_enc: bool
        Whether positional encoding was used or not
    
    Returns:
    str
        The name of the model
    """
    if pos_enc:
        return f"transformer_e{esize}_cs{cs}_nl{nl}_nh{nh}_{epoch}.pt"
    else:
        return f"transformer_e{esize}_cs{cs}_nl{nl}_nh{nh}_nope_{epoch}.pt"

def plot_attention_weights(axs, weights, context):
    """
    Plot the self-attention weights over the context

    Parameters:
    axs: list of matplotlib.axes.Axes
        The axes to plot the attention weights on
    weights: torch.Tensor
        The attention weights for each head. Has shape: (num_heads, seq_len, seq_len)
    context: str
        The context used for the input
    """
    weights = weights.squeeze().cpu().numpy()
    for i in range(weights.shape[0]):
        w = weights[i].T
        w[w == 0] = np.nan
        axs[i].imshow(w, cmap='viridis', interpolation='nearest')
        axs[i].set_title(f'Head {i+1}')
        axs[i].set_xlabel('Input')
        axs[i].set_ylabel('Weights for Input')
        axs[i].set_xticks(range(len(context)))
        axs[i].set_yticks(range(len(context)))
        axs[i].set_xticklabels(list(context))
        axs[i].set_yticklabels(list(context))
        axs[i].set_aspect('auto')
        axs[i].grid(False)

def attention_main(model, prompt, encode, save_name):
    """
    Compute the attention weights for each head and averaged weights for each head and then plot them.
    """

    model = model.to('cpu')
    model.eval()
    mask = torch.nn.Transformer.generate_square_subsequent_mask(len(prompt))
    with torch.no_grad():
        x = torch.tensor(encode(prompt), dtype=torch.long).reshape(-1, 1)
        if isinstance(model, Transformer):
            y_pred, att_weights = model(x, attn_mask=mask, return_att_weights=True)
        else:
            y_pred, att_weights = model(x, mask)
    nrows = int(np.round(att_weights[0].squeeze().shape[0]/2))
    for i, weights in enumerate(att_weights):  # plot the attention weights for each layer
        fig, axs = plt.subplots(nrows,2, figsize=(8, 8))
        plot_attention_weights(axs.flat, weights, prompt)  # plot the attention weights for each head in the layer
        fig.subplots_adjust(left=0.1, right=0.97, top=0.95, bottom=0.1)
        fig.savefig(f'{save_name}_layer{i}.pdf')
    
    avg_weights = torch.stack([w.squeeze().mean(dim=0) for w in att_weights])  # average the attention weights over the heads
    nrows = int(np.round(avg_weights.shape[0]/2))
    fig, axs = plt.subplots(nrows,2, figsize=(8, 8))
    plot_attention_weights(axs.flat, avg_weights, prompt)  # plot the average attention weights for each layer
    for i in range(len(axs.flat)):
        axs.flat[i].set_title(f'Layer {i}')
    fig.subplots_adjust(left=0.1, right=0.97, top=0.95, bottom=0.1)
    fig.savefig(f'{save_name}_avg_weights.pdf')

def find_best_model(dval, vocab_size, esize, cs, nl, nh, width, pos_enc, max_epochs=150):
    """
    Find the best model based on the validation loss. Searches over models saved every 10 epochs up to max_epochs.
    """
    vls = []
    vidx = []

    model = Transformer(vocab_size, embed_size=esize, num_layers=nl, nhead=nh, layer_width=width, 
                        max_len=1024+32, N=512.0, pos_enc=pos_enc)
    mask = torch.nn.Transformer.generate_square_subsequent_mask(cs)
    for i in range(0, max_epochs+1, 10):
        name = model_name(esize, cs, nl, nh, i, pos_enc)
        model.load_state_dict(torch.load(f"weights/{name}", weights_only=True, map_location='cpu'))
        loss = val_loss(dval, model, nn.CrossEntropyLoss(), vocab_size, mask)
        vls.append(loss)
        vidx.append(i)
    plt.plot(vidx, vls)
    plt.savefig(f'val_loss_e{esize}_cs{cs}_nl{nl}_nh{nh}_pe{pos_enc}.png')
    epoch = vidx[np.argmin(vls)]

    model_path = f"weights/{model_name(esize, cs, nl, nh, epoch, pos_enc)}"
    model.load_state_dict(torch.load(model_path, weights_only=True, map_location='cpu'))
    return model, epoch

def plot_attention(prompt, dval, vocab_size, esize, cs, nl, nh, width, encode, max_epochs=150, use_big_model=False):
    """
    Plot the attention weights for the model

    Set use_big_model to True to use the big model. This will load the model from the file model_posenc_True.pth or model_posenc_False.pth
    Set use_big_model to False to use the best model found during training. This will load the model from the file attention_esize_cs_nl_nh_epoch.pth
    """
    
    for pos_enc in [True, False]:  # check attention for models with and without positional encoding
        if use_big_model:
            model = torch.jit.load(f"model_posenc_{pos_enc}.pth")
            save_name = f"attention_big_{pos_enc}"
        else:
            # find the best model
            model, epoch = find_best_model(dval, vocab_size, esize, cs, nl, nh, width, pos_enc, max_epochs=max_epochs)
            save_name = f"attention_{esize}_{cs}_{nl}_{nh}_{epoch}"
            print("Best model found at epoch:", epoch) 

        if pos_enc == False:
            save_name += "_nope"
        attention_main(model, prompt, encode, save_name)

def generate_text(model, prompt, length, context_size, encode, decode):
    """
    Generate text using the model.

    Parameters:
    model: nn.Module
        The model to use for text generation
    prompt: str
        The initial text to use as a prompt
    length: int
        The number of characters to generate
    context_size: int
        The number of characters to use as context. This should be the same as the context_size used in training. We could use the whole generated text as context but this would be slower.
    encode: callable function
        The function to encode text to integers
    decode: callable function
        The function to decode integers to text
    """
    out_idx = []
    model.eval()
    with torch.no_grad():  # don't compute gradients during generation
        mask = torch.nn.Transformer.generate_square_subsequent_mask(len(prompt)+length)
        x = torch.tensor(encode(prompt), dtype=torch.long).unsqueeze(1)  # add batch dimension, i.e., (seq_len,) -> (seq_len, 1)
        out_idx = x.squeeze().tolist()  # add the prompt to the output
        for _ in range(length):
            x = x[-context_size:, :]
            y_pred = model(x, mask[:x.size(0), :x.size(0)]).squeeze()[-1]
            # y_pred = y_pred[-1].argmax().reshape(1, 1)  # greedy decoding. This picks the most likely character at each step.
            probs = F.softmax(y_pred, dim=-1)
            y_pred = torch.multinomial(probs, num_samples=1).unsqueeze(1)  # sampling from the distribution. This adds randomness to the output.
            x = torch.cat((x, y_pred), dim=0)  # add the new prediction to the input
            out_idx = out_idx + [y_pred.squeeze().item()]  # add the new prediction to the output
    out_text = decode(out_idx)  # convert the output indices to text
    return out_text