import torch.nn as nn
import torch 
from torch.autograd import Function

class DomainDiscriminator(nn.Module):
    def __init__(self,in_dim,hidden=256,nd=3):
        super().__init__(); self.net=nn.Sequential(nn.Linear(in_dim,hidden),nn.ReLU(),nn.Linear(hidden,nd))
    def forward(self,x): return self.net(x)

class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx,x,lam): ctx.lam=lam; return x.clone()
    @staticmethod
    def backward(ctx,grad): return grad.neg()*ctx.lam,None

class GradientReversalLayer(nn.Module):
    def __init__(self,lam=1.0): super().__init__(); self.lam=lam
    def forward(self,x): return GradientReversalFunction.apply(x,self.lam)

class FusionAttentionBlock(nn.Module):
    def __init__(self,in_ch,heads=8):
        super().__init__()
        self.attn=nn.MultiheadAttention(embed_dim=in_ch,num_heads=heads)
        self.norm1=nn.LayerNorm(in_ch)
        self.ff=nn.Sequential(nn.Linear(in_ch,in_ch*4),nn.GELU(),nn.Dropout(0.5),nn.Linear(in_ch*4,in_ch))
        self.norm2=nn.LayerNorm(in_ch)
    def forward(self,x):
        B,C,H,W=x.size(); tokens=x.view(B,C,H*W).permute(2,0,1)
        a,_=self.attn(tokens,tokens,tokens); r1=self.norm1(a+tokens)
        f=self.ff(r1); r2=self.norm2(f+r1)
        return r2.permute(1,2,0).view(B,C,H,W)
    
class MultiLevelJointTeacher(nn.Module):
    def __init__(
        self,
        teachers,
        fuse_indices,
        joint_ch=1024,
        heads=8,
        num_attention_layers=3,
        grl_lambda=1.0,
        num_classes=4,
        device='cuda'
    ):
        super().__init__()
        self.n = len(teachers)
        self.joint_ch = joint_ch
        self.teacher_slices = nn.ModuleList()
        self.adapters = nn.ModuleList()

        for idx,t in enumerate(teachers):
            if hasattr(t, 'stem') and hasattr(t, 'features'):
                extractor = nn.Sequential(
                    t.stem,
                    nn.Sequential(*list(t.features.children())[: fuse_indices[idx] + 1])
                )
            else:
                # EfficientNet style
                # use 'features' attr and first few blocks
                extractor = nn.Sequential(
                    t.features[: fuse_indices[idx] + 1]
                )
            extractor.eval()
            for p in extractor.parameters():
                p.requires_grad = False
            self.teacher_slices.append(extractor)

            # dummy input to infer channel dim
            with torch.no_grad():
                dummy = torch.zeros(1, 3, 224, 224).to(device)
                out_feat = extractor(dummy)
            c_out = out_feat.size(1)
            self.adapters.append(nn.Conv2d(c_out, joint_ch, 1))

        # Fusion layers
        self.init_fusion = FusionAttentionBlock(joint_ch * self.n, heads)
        self.projector = nn.Linear(joint_ch * self.n, joint_ch)
        self.fusions = nn.ModuleList([
            FusionAttentionBlock(joint_ch, heads)
            for _ in range(num_attention_layers)
        ])

        # Heads
        self.head = nn.Linear(joint_ch, num_classes)
        self.grl = GradientReversalLayer(grl_lambda)
        self.disc = DomainDiscriminator(joint_ch, hidden=joint_ch, nd=self.n)

    def forward(self, x, return_feats=False):
        feats = []
        for extractor, adapter in zip(self.teacher_slices, self.adapters):
            fmap = extractor(x)           # B x C' x H x W
            adapted = adapter(fmap)       # B x joint_ch x H x W
            pooled = adapted.flatten(2).mean(-1)
            feats.append(pooled)

        V = torch.cat(feats, dim=1)       # B x (joint_ch * n)
        fused = self.init_fusion(V.view(x.size(0), -1, 1, 1))
        v0 = fused.flatten(1)
        proj = self.projector(v0)

        out = proj.unsqueeze(-1).unsqueeze(-1)
        fusion_feats = []
        for block in self.fusions:
            out = block(out)
            fusion_feats.append(out)
        v = out.flatten(2).mean(-1)

        logits = self.head(v)
        dom_logits = self.disc(self.grl(v))

        if return_feats:
            return logits, dom_logits, fusion_feats, v
        return logits, dom_logits
    

class MedViTStudent(nn.Module):
    def __init__(self, model, num_classes, fuse_indices, joint_ch):
        super().__init__()
        # Base MedViT_large without final head
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.backbone = model.to(device)
        self.fuse_idxs = fuse_indices
        # Build 1x1 adapters for each student feature to match joint_ch
        self.adapters = nn.ModuleList()
        dummy = torch.zeros(1, 3, 224, 224).to(device)
        x = self.backbone.stem(dummy)
        for i, block in enumerate(self.backbone.features.children()):
            x = block(x)
            if i in self.fuse_idxs:
                C_s = x.shape[1]
                self.adapters.append(nn.Conv2d(C_s, joint_ch, kernel_size=1))
        # Final classification head
        self.head = nn.Linear(joint_ch, num_classes)

    def forward(self, x):
        feats = []
        x = self.backbone.stem(x)
        adapter_idx = 0
        # Collect & adapt student features at fuse_idxs
        for i, block in enumerate(self.backbone.features.children()):
            x = block(x)
            if i in self.fuse_idxs:
                sf = x  # B x C_s x H x W
                sf_proj = self.adapters[adapter_idx](sf)  # B x joint_ch x H x W
                adapter_idx += 1
                feats.append(sf_proj)
        # Final pooled representation for classification
        v = x.flatten(2).mean(-1)  # B x joint_ch
        out = self.head(v)
        return out, feats