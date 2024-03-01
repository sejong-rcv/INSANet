import torch
import torch.nn as nn
import torch.nn.functional as F


class INSA(nn.Module):
    def __init__(self, 
                 n_iter=1,
                 dim=256,
                 n_head=1,
                 ffn_dim=4
                 ):
    
        super(INSA, self).__init__()
        
        self.dim = dim
        
        self.insa = nn.ModuleList([
            INSA_Transformer(dim=dim,
                             n_head=n_head,
                             ffn_dim=ffn_dim,
                             is_shift=True \
                                 if iter_ % 2 == 1 else False)
            for iter_ in range(n_iter)])
        
        self.param_init()
    
    
    def forward(self, 
                feat_r, 
                feat_t,
                n_window=16,
                **kwargs):
        b, c, h, w = feat_r.shape
        assert self.dim == c
        
        feat_r = feat_r.flatten(-2).permute(0, 2, 1).contiguous()
        feat_t = feat_t.flatten(-2).permute(0, 2, 1).contiguous()
        
        swin_window_mask = self.shift_window_mask(
            size = (h, w),
            window_size_w = w // n_window,
            window_size_h = h // n_window,
            shift_size_w = w // n_window // 2,
            shift_size_h = h // n_window // 2
        )
        
        feat_rt = torch.cat((feat_r, feat_t), dim=0)
        feat_tr = torch.cat((feat_t, feat_r), dim=0)
        
        for insa in self.insa:
            feat_rt = insa(feat_rt, feat_tr,
                           width=w, height=h,
                           mask_attn=swin_window_mask, n_window=n_window)
            
            feat_tr = torch.cat(feat_rt.chunk(chunks=2, dim=0)[::-1], dim=0)
            
        feat_r, feat_t = feat_rt.chunk(chunks=2, dim=0)
        
        feat_r = feat_r.reshape(b, h, w, c).permute(0, 3, 1, 2).contiguous()
        feat_t = feat_t.reshape(b, h, w, c).permute(0, 3, 1, 2).contiguous()
        
        return feat_r, feat_t
    
    
    def shift_window_mask(self,
                          size,
                          window_size_w, 
                          window_size_h,
                          shift_size_w, 
                          shift_size_h):
        # Cite: https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer.py     
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        mask = torch.zeros((1, size[0], size[1], 1)).to(device)
        
        h_windows = (slice(0, -window_size_h),
                        slice(-window_size_h, -shift_size_h),
                        slice(-shift_size_h, None))
        w_windows = (slice(0, -window_size_w),
                        slice(-window_size_w, -shift_size_w),
                        slice(-shift_size_w, None))
        
        count = 0
        for h_window in h_windows:
            for w_window in w_windows:
                mask[:, h_window, w_window, :] = count
                count += 1
        
        b, h, w, c = mask.shape
        n_window = w // window_size_w
        
        mask_windows = mask.reshape(b, n_window, h // n_window, n_window, w // n_window, c
                                    ).permute(0, 1, 3, 2, 4, 5).contiguous()
        mask_windows = mask_windows.reshape(b * n_window ** 2, h // n_window, w // n_window, c
                                            ).reshape(-1, window_size_w * window_size_h)
        mask_attn = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        mask_attn = mask_attn.masked_fill(mask_attn != 0, float(-100.0)).masked_fill(mask_attn == 0, float(0.0))
        
        return mask_attn
    
    
    def param_init(self):
        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)
    
    
class INSA_Transformer(nn.Module):
    # INtra attention and INter attention
    def __init__(self,
                 dim=256,
                 n_head=1,
                 ffn_dim=4,
                 is_shift=True,
                 **kwargs):
        super(INSA_Transformer, self).__init__()
        
        self.intra_attn = Transformer(dim=dim,
                                      n_head=n_head,
                                      is_ffn=False,
                                      ffn_dim=4,
                                      is_shift=is_shift)
        
        self.inter_attn = Transformer(dim=dim,
                                      n_head=n_head,
                                      is_ffn=True,
                                      ffn_dim=4,
                                      is_shift=is_shift)
    
    
    def forward(self,
                feat0, feat1,
                width, height, 
                mask_attn, 
                n_window,
                **kwargs):
        # Intra attention
        feat0 = self.intra_attn(feat0, feat0,
                                width=width, height=height,
                                mask_attn=mask_attn, 
                                n_window=n_window)
        
        # Inter attention
        feat0 = self.inter_attn(feat0, feat1,
                                width=width, height=height,
                                mask_attn=mask_attn, 
                                n_window=n_window)
        
        return feat0
        
        
class Transformer(nn.Module):
    # Cite: https://github.com/haofeixu/gmflow/blob/main/gmflow/transformer.py
    # Transformer (Q,K,V)
    def __init__(self,
                 dim=256,
                 n_head=1,
                 is_ffn=True,
                 ffn_dim=4,
                 is_shift=True,
                 **kwargs):
        super(Transformer, self).__init__()
        
        self.is_ffn = is_ffn
        self.ffn_dim = ffn_dim
        self.is_shift = is_shift
        
        # Multi-Head Attention (MHA)
        self.q_emb = nn.Linear(dim, dim, bias=False)
        self.k_emb = nn.Linear(dim, dim, bias=False)            
        self.v_emb = nn.Linear(dim, dim, bias=False)
        
        self.fc = nn.Linear(dim, dim, bias=False)
        self.norm = nn.LayerNorm(dim)

        # Feed Forward Network (FFN) that only apply after Inter attention
        if is_ffn:
            self.mlp = nn.Sequential(
                nn.Linear(dim*2, dim*2*ffn_dim, bias=False),
                nn.GELU(),
                nn.Linear(dim*2*ffn_dim, dim, bias=False)
                )
                
            self.ffn_norm = nn.LayerNorm(dim)
           
            
    def forward(self,
                feat0, feat1,
                width, height,
                mask_attn, 
                n_window):
        q, k, v = feat0, feat1, feat1

        q_proj = self.q_emb(q)
        k_proj = self.k_emb(k)
        v_proj = self.v_emb(v)
        
        attn_feat = self.window_attention(q_proj, k_proj, v_proj,
                                          width=width, height=height,
                                          n_window=n_window,
                                          mask_attn=mask_attn)
        
        attn_feat = self.norm(self.fc(attn_feat))
        
        if self.is_ffn:
            attn_feat = self.mlp(torch.cat([
                feat0, attn_feat], dim=-1))
            attn_feat = self.ffn_norm(attn_feat)
            
        return feat0 + attn_feat
        
    
    def window_partition(self,
                         feature,
                         n_window):
        b, h, w, c = feature.shape
        
        feature = feature.reshape(b, n_window, h // n_window, n_window, w // n_window,
                                  c).permute(0, 1, 3, 2, 4, 5).contiguous()
        feature = feature.reshape(b * n_window ** 2, h // n_window, w // n_window, c)
        
        return feature
        
        
    def window_merge(self,
                     feature,
                     n_window):
        b, h, w, c = feature.shape
        b_ = b // n_window ** 2
        
        feature = feature.view(b_, n_window, n_window, h, w, c)
        feat_merge = feature.permute(0, 1, 3, 2, 4, 5).contiguous().reshape(
            b_, n_window * h, n_window * w, c)
        
        return feat_merge
    
    
    def window_attention(self,
                         q, k, v,
                         width, height,
                         n_window, 
                         mask_attn):
        # Cite: https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer.py 
        assert q.dim() == k.dim() == v.dim() == 3
        
        assert width is not None and height is not None
        assert q.size(1) == width * height
        
        b, _, c = q.shape
        
        b_ = b * n_window ** 2
        window_size_w = width // n_window
        window_size_h = height // n_window
        
        q = q.view(b, height, width, c)
        k = k.view(b, height, width, c)
        v = v.view(b, height, width, c)
        
        scale_factor = c ** 0.5
        
        if self.is_shift:
            assert mask_attn is not None
            shift_size_w = window_size_w // 2
            shift_size_h = window_size_h // 2
            
            q = torch.roll(q, shifts=(-shift_size_h, -shift_size_w), dims=(1, 2))
            k = torch.roll(k, shifts=(-shift_size_h, -shift_size_w), dims=(1, 2))
            v = torch.roll(v, shifts=(-shift_size_h, -shift_size_w), dims=(1, 2))
            
        q = self.window_partition(q, n_window=n_window)
        k = self.window_partition(k, n_window=n_window)
        v = self.window_partition(v, n_window=n_window)
        
        scores = torch.matmul(q.view(b_, -1, c),
                              k.view(b_, -1, c).permute(0, 2, 1)
                              ) / scale_factor
        
        if self.is_shift:
            scores += mask_attn.repeat(b, 1, 1)
            
        attn_scores = torch.softmax(scores, dim=-1)
        
        out_feat = torch.matmul(attn_scores, v.view(b_, -1, c))
        out_feat = self.window_merge(out_feat.view(b_, height // n_window, width // n_window, c),
                                     n_window=n_window)
        
        if self.is_shift:
            out_feat = torch.roll(out_feat, shifts=(shift_size_h, shift_size_w), dims=(1, 2))
        
        return out_feat.reshape(b, -1, c)
