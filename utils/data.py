import torch
import numpy as np

from scipy.special import softmax


def generate_x(example_num, x_dim, batch_num=1, data_range=1, xi=0, device="cuda:0"):
    mean = torch.zeros(x_dim, device=device)
    cov = torch.eye(x_dim, device=device) * data_range + xi
    dist = torch.distributions.MultivariateNormal(mean, cov)
    x_all = dist.sample((batch_num, example_num))
    return x_all  # b * t * c

def generate_y(x_all, weight_range=1, noise=None, few_shot=False, 
              choose_elements=1, assign_dims=None,device="cuda:0"):

    random_weight = torch.randn(x_all.shape[0], 1, x_all.shape[-1], 
                                device=device) * weight_range
    
    if few_shot: # for sparse linear regression
        if assign_dims is not None:
            weight_mask = torch.ones_like(random_weight, dtype=torch.bool)
            for i in assign_dims:
                weight_mask[:,:,i] = False
        else:
            assert choose_elements <= x_all.shape[-1]
            weight_mask = torch.randn(x_all.shape[0], 1, x_all.shape[-1], 
                                    device=device) * weight_range
            weight_mask_sort = torch.sort(weight_mask, dim=-1)[0]
            weight_mask = (weight_mask > weight_mask_sort[:,:,choose_elements-1].unsqueeze(-1))
        random_weight[weight_mask] = 0
        
    y = (random_weight * x_all).sum(dim=-1, keepdim=True)
    if noise is not None:
        noise_data = torch.randn(*y.shape, device=device) * noise
        y += noise_data
    return y, random_weight.squeeze()

# def conduct_gd(x_context, y_context, eta , w0 = None, steps = 1):
#     device = x_context.device
#     b,t,c = x_context.size()
#     if w0 is None:
#         wi = torch.zeros((b,c,1), device = device)
#     else:
#         wi = w0
#     x_context_t = x_context.transpose(1, 2)  # b * c * t
#     w_cot = [wi.transpose(1, 2).clone()]
#     for _ in range(steps):
#         wi -= eta / t * (x_context_t @ (x_context @ wi - y_context))
#         # b * c * t @ b * t * c @ b * c * 1 => b * c * 1
#         w_cot.append(wi.transpose(1, 2).clone())
    
#     w_cot = torch.concat(w_cot, dim = 1)
#     return w_cot

def combine_input(x_context, y_context, w_cot):
    # x_context: b * t * c
    # y_context: b * t * 1
    # w_cot: b * k * c
    b,k,c = w_cot.size()
    t = x_context.size(1)
    device = x_context.device
    wcot_paddings1 = torch.zeros((b, k, c + 1), device = device)
    wcot_paddings2 = torch.ones((b, k, 1), device = device)
    xy_paddings = torch.zeros((b, t, c + 1), device = device)
    xy_context = torch.concat(
        [x_context,y_context,xy_paddings], dim = -1
    )
    wcot_input = torch.concat(
        [wcot_paddings1,w_cot,wcot_paddings2], dim = -1
    )
    input_seq = torch.concat(
        [xy_context, wcot_input], dim = -2
    )
    return input_seq # b * (t + k) * (2 * c + 2)

def conduct_gd_noise(x_context, y_context, eta , w0 = None, steps = 1, lbda = 0, gd_noise = 0, gd_noise_func = None):
    device = x_context.device
    b,t,c = x_context.size()
    if w0 is None:
        wi = torch.zeros((b,c,1), device = device)
    else:
        wi = w0
    x_context_t = x_context.transpose(1, 2)  # b * c * t
    w_cot = [wi.transpose(1, 2).clone()]
    
    for i in range(steps):
        if gd_noise_func is not None:
            wi -= eta * ((x_context_t @ (x_context @ wi - y_context)) + lbda * wi + gd_noise_func(gd_noise,wi,device)) 
        else:
            wi -= eta * ((x_context_t @ (x_context @ wi - y_context)) + lbda * wi)
        w_cot.append(wi.transpose(1, 2).clone())
    w_cot = torch.concat(w_cot, dim = 1)
    return w_cot



def conduct_gd_discrete(x_context, y_context, eta, w0=None, steps=1, select_method="sample", choose_elements = 1):
    device = x_context.device
    b, t, c = x_context.size()
    encode_list = 2**torch.arange(c,dtype=torch.int64, device = device)
    encode_list = encode_list[None,:]
    if w0 is None:
        wi = torch.zeros((b, c, 1), device=device)
    else:
        wi = w0

    x_context_t = x_context.transpose(1, 2)  # b * c * t
    w_cot = []
    
    for i in range(steps):
        # Gradient update
        wi -= eta * (x_context_t @ (x_context @ wi - y_context))
        wi = wi.clip(min=1e-10)  # Prevents division by zero
        wi /= wi.sum(dim=1, keepdim=True)  # Normalize weights
        if select_method == "sample":
            indices = torch.multinomial(wi.squeeze(-1), num_samples=choose_elements, replacement=False)
        elif select_method == "greedy":
            indices = wi.squeeze(-1).topk(choose_elements, dim=-1).indices  # Top-k greedy selection
        else:
            raise NotImplementedError("Unknown selection method: {}".format(select_method))
        assert indices.size(-1) == choose_elements
        wi = torch.zeros((b, c), device=device)
        wi[torch.arange(b, device=device).unsqueeze(1), indices] = 1
        wi_encode = (encode_list * wi.to(torch.int64)).sum(dim = -1).cpu().numpy()
        wi = wi.unsqueeze(-1)  # Restore original shape (b, c, 1)
        w_cot.append(wi_encode[:, None].copy())
    w_cot = np.concatenate(w_cot, axis = 1)
    return w_cot