import torch
import torch.nn as nn
from timm.models.vision_transformer import Attention, Mlp

class WindowAttBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, num, size, mlp_ratio=4.0):
        super().__init__()
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.num, self.size = num, size

        self.nnorm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.nattn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, attn_drop=0.1, proj_drop=0.1)
        self.nnorm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.nmlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=0.1)

        self.snorm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.sattn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, attn_drop=0.1, proj_drop=0.1)
        self.snorm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.smlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=0.1)

    def forward(self, x):
        B,T,_,D = x.shape
        # P: ptach num and N: patch size
        P, N = self.num, self.size
        assert self.num * self.size == _
        x = x.reshape(B, T, P, N, D)

        # depth attention
        qkv = self.snorm1(x.reshape(B*T*P,N,D))
        x = x + self.sattn(qkv).reshape(B,T,P,N,D)
        x = x + self.smlp(self.snorm2(x))
        
        # breadth attention
        qkv = self.nnorm1(x.transpose(2,3).reshape(B*T*N,P,D))
        x = x + self.nattn(qkv).reshape(B,T,N,P,D).transpose(2,3)
        x = x + self.nmlp(self.nnorm2(x))
         
        return x.reshape(B,T,-1,D)

class PatchSTG(nn.Module):
    def __init__(self, tem_patchsize, tem_patchnum,
                        node_num, spa_patchsize, spa_patchnum,
                        tod, dow,
                        layers, factors,
                        input_dims, node_dims, tod_dims, dow_dims,
                        ori_parts_idx, reo_parts_idx, reo_all_idx
                ):
        super(PatchSTG, self).__init__()
        self.node_num = node_num
        self.ori_parts_idx, self.reo_parts_idx = ori_parts_idx, reo_parts_idx
        self.reo_all_idx = reo_all_idx
        self.tod, self.dow = tod, dow

        # model_dims = input_emb + spa_emb + tem_emb
        dims = input_dims + tod_dims + dow_dims + node_dims

        # spatio-temporal embedding -> section 4.1 in paper
        # input_emb
        self.input_st_fc = nn.Conv2d(in_channels=3, out_channels=input_dims, kernel_size=(1, tem_patchsize), stride=(1, tem_patchsize), bias=True)
        # spa_emb
        self.node_emb = nn.Parameter(
                torch.empty(node_num, node_dims))
        nn.init.xavier_uniform_(self.node_emb)
        # tem_emb
        self.time_in_day_emb = nn.Parameter(
                torch.empty(tod, tod_dims))
        nn.init.xavier_uniform_(self.time_in_day_emb)
        self.day_in_week_emb = nn.Parameter(
                torch.empty(dow, dow_dims))
        nn.init.xavier_uniform_(self.day_in_week_emb)

        # dual attention encoder -> section 4.3 in paper, factors for merging the leaf nodes of KDTree
        self.spa_encoder = nn.ModuleList([
            WindowAttBlock(dims, 1, spa_patchnum//factors, spa_patchsize*factors, mlp_ratio=1) for _ in range(layers)
        ])

        # projection decoder -> section 4.4 in paper
        self.regression_conv = nn.Conv2d(in_channels=tem_patchnum*dims, out_channels=tem_patchsize*tem_patchnum, kernel_size=(1, 1), bias=True)

    def forward(self, x, te):
        # x: [B,T,N,1] input traffic
        # te: [B,T,N,2] time information

        # spatio-temporal embedding -> section 4.1 in paper
        embeded_x = self.embedding(x, te)
        rex = embeded_x[:,:,self.reo_all_idx,:] # select patched points

        # dual attention encoder -> section 4.3 in paper
        for block in self.spa_encoder:
            rex = block(rex)

        orginal = torch.zeros(rex.shape[0],rex.shape[1],self.node_num,rex.shape[-1]).to(x.device)
        orginal[:,:,self.ori_parts_idx,:] = rex[:,:,self.reo_parts_idx,:] # back to the original indices

        # projection decoder -> section 4.4 in paper
        pred_y = self.regression_conv(orginal.transpose(2,3).reshape(orginal.shape[0],-1,orginal.shape[-2],1))

        return pred_y # [B,T,N,1]

    def embedding(self, x, te):
        b,t,n,_ = x.shape

        # input traffic + time of day + day of week as the input signal
        x1 = torch.cat([x,(te[...,0:1]/self.tod),(te[...,1:2]/self.dow)], -1).float()
        input_data = self.input_st_fc(x1.transpose(1,3)).transpose(1,3)
        t, d = input_data.shape[1], input_data.shape[-1]        

        # cat time of day embedding
        t_i_d_data = te[:, -input_data.shape[1]:, :, 0]
        input_data = torch.cat([input_data, self.time_in_day_emb[(t_i_d_data).type(torch.LongTensor)]], -1)

        # cat day of week embedding
        d_i_w_data = te[:, -input_data.shape[1]:, :, 1]
        input_data = torch.cat([input_data, self.day_in_week_emb[(d_i_w_data).type(torch.LongTensor)]], -1)

        # cat spatial embedding
        node_emb = self.node_emb.unsqueeze(0).unsqueeze(1).expand(b, t, -1, -1)
        input_data = torch.cat([input_data, node_emb], -1)

        return input_data