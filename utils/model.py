import torch
import torch.nn as nn

class SimpleAttn(nn.Module):
    def __init__(self, input_size, head_num = None,  layer_num = 1, context_length = None):
        super(SimpleAttn, self).__init__()
        if head_num is None:
            head_num = 1
        self.head_num = head_num
        # initialize parameter
        self.w_ov = nn.Parameter(torch.randn(input_size,input_size*self.head_num)*0.002/layer_num) # input_size * (head_num * input_size)
        self.w_qk = nn.Parameter(torch.randn(input_size,input_size*self.head_num)*0.002/layer_num) # input_size * (head_num * input_size)
        # self.w_ov = nn.Parameter(torch.concat([torch.eye(input_size) for _ in range(self.head_num)], dim = -1))# input_size * (head_num * input_size)
        # self.w_qk = nn.Parameter(torch.concat([torch.eye(input_size) for _ in range(self.head_num)], dim = -1)) # input_size * (head_num * input_size)
        self.context_length = context_length
    
    def forward(self, input_seq, **kwargs):
        assert input_seq.dim() == 3 # b * t * c
        b,t,c = input_seq.size()
        Q = torch.concat((input_seq,)*self.head_num,dim = -1).reshape(b,t,-1,c) # b * t * (c * h) -> b * t * h * c
        K = (input_seq @ self.w_qk).reshape(b,t,-1,c) # b * t * (c * h) -> b * t * h * c
        V = (input_seq @ self.w_ov).reshape(b,t,-1,c) # b * t * (c * h) -> b * t * h * c
        
        Q = Q.transpose(1,2) # b * t * h * c -> b * h * t * c
        K = K.transpose(1,2) # b * t * h * c -> b * h * t * c
        V = V.transpose(1,2) # b * t * h * c -> b * h * t * c
        A = Q @ K.transpose(-1,-2) # A : b * h * t * t
        if self.context_length is not None:
            A[self.context_length: , self.context_length :] = 0
        out = A @ V # b * h * t * c 
        out = out.transpose(1, 2) # b * h * t * c -> b * t * h * c 
        return out # b * t * h * c 