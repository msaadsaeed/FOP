"""
"Fusion and Orthogonal Projection for Improved Face-Voice Association"
Muhammad Saad Saeed and Muhammad Haris Khan and Shah Nawaz and Muhammad Haroon Yousaf and Alessio Del Bue
ICASSP 2022
"""

import torch
import torch.nn as nn

'''
Gated Multi-Modal Fusion
'''

class GatedFusion(nn.Module):
    def __init__(self, face_input, voice_input, embed_dim_in, mid_att_dim, emb_dim_out):
        super(GatedFusion, self).__init__()
        self.linear_face = nn.Sequential()
        self.linear_voice = nn.Sequential()
        self.final_transform = nn.Sequential()
        self.attention = nn.Sequential(
            Forward_Block(embed_dim_in*2, mid_att_dim),
            nn.Linear(mid_att_dim, emb_dim_out)
            )

    def forward(self, face_input, voice_input):
        concat = torch.cat((face_input, voice_input), dim=1)
        attention_out = torch.sigmoid(self.attention(concat))
        face_trans = torch.tanh(self.linear_face(face_input))
        voice_trans = torch.tanh(self.linear_voice(voice_input))
        
        out = face_trans * attention_out + (1.0 - attention_out) * voice_trans
        out = self.final_transform(out)
        
        return out, face_trans, voice_trans

class Forward_Block(nn.Module):
    
    def __init__(self, input_dim=128, output_dim=128, p_val=0.0):
        super(Forward_Block, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.Dropout(p=p_val)
        )
    def forward(self, x):
        return self.block(x)


def make_fc_1d(f_in, f_out):
    return nn.Sequential(nn.Linear(f_in, f_out), 
                        nn.BatchNorm1d(f_out),
                        nn.ReLU(inplace=True),
                        nn.Dropout(p=0.5))

'''
Linear Weighted Addition
'''

class LinearWeightedAvg(nn.Module):
    def __init__(self, face_feat_dim, voice_feat_dim):
        super(LinearWeightedAvg, self).__init__()
        self.weight1 = nn.Parameter(torch.rand(1, device='cuda')).requires_grad_()
        self.weight2 = nn.Parameter(torch.rand(1, device='cuda')).requires_grad_() 
    def forward(self, face_feat, voice_feat):
        return self.weight1 * face_feat + self.weight2 * voice_feat, face_feat, voice_feat

'''
Embedding Extraction Module
'''        

class EmbedBranch(nn.Module):
    def __init__(self, feat_dim, embedding_dim):
        super(EmbedBranch, self).__init__()
        self.fc1 = make_fc_1d(feat_dim, embedding_dim).cuda()

    def forward(self, x):
        x = self.fc1(x)
        x = nn.functional.normalize(x) 
        return x

'''
Main Module
'''

class FOP(nn.Module):
    def __init__(self, args, face_feat_dim, voice_feat_dim):
        super(FOP, self).__init__()
        
        self.voice_branch = EmbedBranch(voice_feat_dim, args.dim_embed)
        self.face_branch = EmbedBranch(face_feat_dim, args.dim_embed)
        
        if args.fusion == 'linear':
            self.fusion_layer = LinearWeightedAvg(args.dim_embed, args.dim_embed)
        elif args.fusion == 'gated':
            self.fusion_layer = GatedFusion(face_feat_dim, voice_feat_dim, args.dim_embed, 128, args.dim_embed)
        
        self.logits_layer = nn.Linear(args.dim_embed, 901)

        if args.cuda:
            self.cuda()

    def forward(self, faces, voices):
        voices = self.voice_branch(voices)
        faces = self.face_branch(faces)
        feats, faces, voices = self.fusion_layer(faces, voices)
        logits = self.logits_layer(feats)
        
        return [feats, logits], faces, voices
    
    def train_forward(self, faces, voices, labels):
        
        comb, face_embeds, voice_embeds = self(faces, voices)
        return comb, face_embeds, voice_embeds
