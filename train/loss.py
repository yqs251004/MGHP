import torch
from torch import nn

def register_activation_hook(model):
    activations = {}
    handles = []
    for name, module in model.named_modules():
        module.name = name
        def _hook(mod, __, val):
            activations[mod.name] = val
        handles.append(module.register_forward_hook(_hook))
    return activations, handles


class MMD_loss(nn.Module):
    def __init__(self, kernel_mul = 2.0, kernel_num = 5):
        super(MMD_loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        return
    
    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0])+int(target.size()[0])
        total = torch.cat([source, target], dim=0)
 
        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)
 
    def forward(self, source, target, xy_only=False):
        batch_size = int(source.size()[0])
        kernels = self.guassian_kernel(source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        loss = torch.mean(XX + YY - XY -YX)
        return loss


def masked_token_ce_loss(logits, labels, mask=None):
    if mask is None:
        mask = (labels != -100)

    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    shift_label_mask = mask[..., 1:].contiguous()

    # 把 mask 的位置设为 -100，配合 ignore_index
    shift_labels = shift_labels.masked_fill(~shift_label_mask, -100)

    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
    loss = loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
    )
    return loss

def weighted_ce_loss(logits, labels, mask=None):
    if mask is None:
        mask = (labels != -100)
        
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    shift_label_mask = mask[..., 1:].contiguous()

    # 把 mask 的位置设为 -100，配合 ignore_index
    shift_labels = shift_labels.masked_fill(~shift_label_mask, -100)

    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
    loss = loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
    )
    loss = loss.view(shift_labels.size())
    # we use a simple weighting strategy where the first response token has weight 1, and the rest tokens have weight 0.5
    response_start = (shift_labels != -100).int().argmax(dim=1)
    weights = torch.ones_like(shift_labels, dtype=torch.float)
    for i in range(weights.size(0)):
        start = response_start[i].item()
        weights[i, start:start + 3] = 2.0
        weights[i, start + 3:] = 1.0
    loss = loss * weights

    valid_weights = weights * (shift_labels != -100).float()
    token_counts = (shift_labels != -100).sum(dim=1)  # 每个样本的有效 token 数量
    token_counts = token_counts.masked_fill(token_counts == 0, 1)  # 避免除以零
    loss = (loss.sum(dim=1) / valid_weights.sum(dim=1).clamp_min(1)).mean()  # 先对每个样本求平均，再对 batch 求平均
    return loss

def gaussian_kernel_loss(embeddings, temperature=2):
    """ Select the last token of the prompt and blur the connections
     Args:
        embeddings (torch.Tensor): The embeddings of shape (batch_size, hidden_size)
    """
    batch_size, _ = embeddings.size()
    embeddings = embeddings.to(torch.float32)

    # compute the l2 norm between each pair of embeddings
    # D = torch.cdist(embeddings, embeddings, p=2).pow(2)

    # compute the dot product between each pair of embeddings
    D = torch.matmul(embeddings, embeddings.t())
    norm = torch.norm(embeddings, dim=1, keepdim=True)
    D = D / (norm @ norm.t())
    D = abs(D) / temperature
    # A = torch.log(torch.sigmoid(D / temperature) + 1e-8)
    I = torch.ones(batch_size, batch_size).to(embeddings.device)
    A = torch.log(I - D)
    loss = -(torch.sum(A) - torch.sum(torch.diag(A))) / (batch_size * (batch_size - 1))
    return loss
        

def rep_noise_loss(
    model,
    harmful_batch,
    harmless_batch,
    activations,
    beta=0.001,
    alpha=1,
):
    """ Calculate the representation noise loss

    Args:
        model (Pytorch Model): The model to calculate the loss, i.e. an LLM loaded with Huggingface Transformers
        harmful_batch (Dataloader): the paired harmful batch
        harmless_batch (Dataloader): the paired harmless batch
        beta (float, optional): _description_. Defaults to 0.001.
        alpha (int, optional): _description_. Defaults to 1.
    """
    mmd_loss = MMD_loss()

    harmful_outputs = model(harmful_batch['input_ids'], attention_mask=harmful_batch['attention_mask'], output_hidden_states=True)
    harmful_activations = []
    for i in range(len(model.base_model.layers)):
        harmful_activations.append(activations[f'model.layers.{i}.mlp'])
    mask = (harmful_batch['labels'] != -100)

    noise_loss = 0
    for i, hidden in enumerate(harmful_activations):
        hiddens_mask = mask.unsqueeze(-1).expand(hidden.size()).to(hidden.device)
        hiddens = hidden * hiddens_mask
        gaussian = torch.randn_like(hiddens).to(hidden.device) * hiddens_mask
        noise_loss += mmd_loss(hiddens.view(hiddens.size(0), -1), gaussian.view(gaussian.size(0), -1))
    noise_loss /= len(harmful_activations) # len(layer_idxs)

    harmful_losses = masked_token_ce_loss(
        harmful_outputs.logits,
        harmful_batch['labels'],
        mask
    )

    output_embeddings = model.get_output_embeddings()
    norm = model.base_model.norm
    for i, h in enumerate(harmful_outputs.hidden_states):
        out = output_embeddings(norm(h))
        loss = masked_token_ce_loss(
            out,
            harmful_batch['labels'],
            mask
        )
        harmful_losses += loss
    harmful_losses = harmful_losses / len(harmful_outputs.hidden_states) + 1

    mask = (harmless_batch['labels'] != -100)
    harmless_outputs = model(harmless_batch['input_ids'], attention_mask=harmless_batch['attention_mask'],output_hidden_states=True)
    harmless_losses = masked_token_ce_loss(
        harmless_outputs.logits,
        harmless_batch['labels'],
        mask
    )

    print(f"Harmful Losses: {harmful_losses.item():.4f}, Harmless Losses: {harmless_losses.item():.4f}, Noise Loss: {noise_loss.item():.4f}")
    loss = harmless_losses + beta * noise_loss - alpha * torch.log(harmful_losses)

    return loss, harmless_losses, noise_loss, harmful_losses

def contrastive_loss(
    model,
    model_name,
    harmful_batch,
    harmless_batch,
    activations,
    beta=0.001,
    alpha=1,
):
    """ Calculate the representation noise loss

    Args:
        model (Pytorch Model): The model to calculate the loss, i.e. an LLM loaded with Huggingface Transformers
        harmful_batch (Dataloader): the paired harmful batch
        harmless_batch (Dataloader): the paired harmless batch
        beta (float, optional): _description_. Defaults to 0.001.
        alpha (int, optional): _description_. Defaults to 1.
    """

    harmful_outputs = model(harmful_batch['input_ids'], attention_mask=harmful_batch['attention_mask'], output_hidden_states=True)
 
    # select the last layer outputs
    # harmful_activations = harmful_outputs.hidden_states[-1] # shape: (batch_size, seq_len, hidden_size)
    mask = (harmful_batch['labels'] != -100)
    # take the first location of the mask with true value
    idx = mask.int().argmax(dim=1)  # shape: (batch_size,)
    if model_name == 'qwen':
        idx = idx - 6  # for qwen models the last prompt token is 6 tokens before the first non-masked token
    else:
        raise NotImplementedError("Currently only qwen model is supported for contrastive loss.")

    # harmful_activations = harmful_activations[torch.arange(harmful_activations.size(0)), idx]
    # noise_loss = gaussian_kernel_loss(harmful_activations, temperature=2.0)

    harmful_losses = masked_token_ce_loss(
        harmful_outputs.logits,
        harmful_batch['labels'],
        mask
    )

    noise_loss = 0
    output_embeddings = model.get_output_embeddings()
    norm = model.base_model.norm
    for i, h in enumerate(harmful_outputs.hidden_states):
        harmful_activations = h[torch.arange(h.size(0)), idx]
        noise_loss += gaussian_kernel_loss(harmful_activations, temperature=2.0)

        out = output_embeddings(norm(h))
        loss = masked_token_ce_loss(
            out,
            harmful_batch['labels'],
            mask
        )
        harmful_losses += loss

    noise_loss = noise_loss / len(harmful_outputs.hidden_states)
    harmful_losses = harmful_losses / len(harmful_outputs.hidden_states) + 1

    mask = (harmless_batch['labels'] != -100)
    harmless_outputs = model(harmless_batch['input_ids'], attention_mask=harmless_batch['attention_mask'],output_hidden_states=True)
    harmless_losses = masked_token_ce_loss(
        harmless_outputs.logits,
        harmless_batch['labels'],
        mask
    )

    print(f"Harmful Losses: {harmful_losses.item():.4f}, Harmless Losses: {harmless_losses.item():.4f}, Noise Loss: {noise_loss.item():.4f}")
    loss = harmless_losses + beta * noise_loss - alpha * torch.log(harmful_losses)

    return loss, harmless_losses, noise_loss, harmful_losses
