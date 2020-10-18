import torch

class IAddMM(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, types, W, b):
        """
        x : float[B, ..., in_size]
        types : (0..n_types)[B]
        W : float[n_types, out_size, in_size]
        b : float[n_types, out_size]
        return : float[B, ..., out_size] = xW[types] + b[types]
        """
        B = x.shape[0]
        mid_size = x.shape[1:-1]
        mid_dims = len(mid_size)
        in_size = x.shape[-1]
        out_size = W.shape[1]
        
        y = torch.zeros(B, *mid_size, out_size, device=x.device)
        
        for t in range(types.max().item() + 1):
            t_mask = (types == t)
            if not t_mask.any():
                continue
            W_t = W[t]
            b_t = b[t]
            x_t = x[t_mask].unsqueeze(-1)
            y_t = (W_t @ x_t).squeeze(-1) + b_t
            y[t_mask] = y_t
            
        ctx.save_for_backward(x, types, W, b)
        return y
    
    @staticmethod
    def backward(ctx, grad_y):
        x, types, W, b = ctx.saved_tensors
        grad_x = torch.zeros_like(x)
        grad_W = torch.zeros_like(W)
        grad_b = torch.zeros_like(b)
        
        for t in range(types.max().item() + 1):
            t_mask = (types == t)
            if not t_mask.any():
                continue
            grad_y_t = grad_y[t_mask]
            grad_x[t_mask] = grad_y_t @ W[t]
            grad_W[t] = grad_y_t.view(-1, grad_y_t.shape[-1]).T @ x[t_mask].view(-1, x.shape[-1])
            grad_b[t] = grad_y_t.view(-1, b.shape[-1]).sum(0)
        return grad_x, None, grad_W, grad_b
    
def iaddmm(x, types, W, b):
    return IAddMM.apply(x, types, W, b)
    
class IMM(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, types, W):
        """
        x : float[B, ..., in_size]
        types : (0..n_types)[B]
        W : float[n_types, out_size, in_size]
        return : float[B, ..., out_size] = xW[types] + b[types]
        """
        B = x.shape[0]
        mid_size = x.shape[1:-1]
        mid_dims = len(mid_size)
        in_size = x.shape[-1]
        out_size = W.shape[1]
        
        y = torch.zeros(B, *mid_size, out_size, device=x.device)
        
        for t in range(types.max().item() + 1):
            t_mask = (types == t)
            if not t_mask.any():
                continue
            W_t = W[t]
            x_t = x[t_mask].unsqueeze(-1)
            y_t = (W_t @ x_t).squeeze(-1)
            y[t_mask] = y_t
            
        ctx.save_for_backward(x, types, W)
        return y
    
    @staticmethod
    def backward(ctx, grad_y):
        x, types, W = ctx.saved_tensors
        grad_x = torch.zeros_like(x)
        grad_W = torch.zeros_like(W)
        
        for t in range(types.max().item() + 1):
            t_mask = (types == t)
            if not t_mask.any():
                continue
            grad_y_t = grad_y[t_mask]
            grad_x[t_mask] = grad_y_t @ W[t]
            grad_W[t] = grad_y_t.view(-1, grad_y_t.shape[-1]).T @ x[t_mask].view(-1, x.shape[-1])
        return grad_x, None, grad_W
    
def imm(x, types, W):
    return IMM.apply(x, types, W)

class TLayerNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, types, w, b):
        ctx.save_for_backward(x, types, w, b)
        x = (x - x.mean(-1, keepdim=True)) / (x.var(-1, keepdim=True, unbiased=False) + 1e-5).sqrt()
        w = w[types][(slice(None),) + (None,) * (x.dim() - 2)]
        b = b[types][(slice(None),) + (None,) * (x.dim() - 2)]
        return x * w + b
    
    @staticmethod
    def backward(ctx, grad_y):
        x, types, w, b = ctx.saved_tensors
        types_expanded = types[(slice(None),) + (None,) * (x.dim() - 1)].expand_as(grad_y)
        
        delta = x - x.mean(-1, keepdim=True)
        v = (x.var(-1, keepdim=True, unbiased=False) + 1e-5).sqrt()
        x_norm = delta / v
        n = x.shape[-1]
        
        grad_w = torch.zeros(w.shape[0], *x.shape[1:], device=w.device)
        grad_w.scatter_add_(0, types_expanded, grad_y * x_norm)
        grad_w = torch.einsum('i...j->ij', grad_w)
        
        grad_b = torch.zeros(b.shape[0], *x.shape[1:], device=b.device)
        grad_b.scatter_add_(0, types_expanded, grad_y)
        grad_b = torch.einsum('i...j->ij', grad_b)
        
        w_t = w[types][(slice(None),) + (None,) * (x.dim() - 2)].expand_as(grad_y)
        grad_x = grad_y * w_t / v
        grad_x += -1 / n / v * torch.einsum('...ij,...ij->...i', grad_y, w_t)[..., None]
        grad_x += -1 / n / (v ** 3) * delta * torch.einsum('...ij,...ij,...ij->...i', grad_y, delta, w_t)[..., None]
        
        return grad_x, None, grad_w, grad_b
    
def typed_layer_norm(x, types, w, b):
    return TLayerNorm.apply(x, types, w, b)
