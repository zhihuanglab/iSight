import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoModelForZeroShotImageClassification, AutoProcessor, AutoConfig
import random

class SelfAttention(nn.Module):
    def __init__(self, L):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(L, L)
        self.key = nn.Linear(L, L)
        self.value = nn.Linear(L, L)
        self.scale = 1. / (L ** 0.5)

    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        attn_weights = F.softmax(q @ k.transpose(-2, -1) * self.scale, dim=-1)
        return attn_weights @ v

# class Attn_Net_Gated(nn.Module):
#     def __init__(self, L=1024, D=256, dropout=0.25, n_classes=1):
#         super(Attn_Net_Gated, self).__init__()
        
#         # Self-attention layer for patch contextual awareness
#         self.self_attention = SelfAttention(L)
        
#         # Attention branches with normalization
#         self.attention_a = nn.Sequential(
#             nn.Linear(L, D),
#             nn.Tanh(),
#             nn.LayerNorm(D)
#         )
#         self.attention_b = nn.Sequential(
#             nn.Linear(L, D),
#             nn.Sigmoid(),
#             nn.LayerNorm(D)
#         )
        
#         # Dynamic gating mechanism to adaptively weight attention branches
#         self.dynamic_gate = nn.Linear(L, 2)  # Outputs weights for `attention_a` and `attention_b`

#         if dropout:
#             self.attention_a.add_module("Dropout", nn.Dropout(dropout))
#             self.attention_b.add_module("Dropout", nn.Dropout(dropout))

#         # Final attention scoring layer
#         self.attention_c = nn.Linear(D, n_classes)
        
#         # Learnable scaling parameter for sharper attention scores
#         self.temperature = nn.Parameter(torch.tensor(1.0))
        
#     def forward(self, x):
#         # Apply self-attention for better patch-to-patch contextual understanding
#         x = self.self_attention(x)
        
#         # Pass through attention branches
#         a = self.attention_a(x)
#         b = self.attention_b(x)
        
#         # Dynamic gating: learnable weights for combining `a` and `b`
#         gate_weights = F.softmax(self.dynamic_gate(x), dim=-1)
#         A = gate_weights[:, 0].unsqueeze(-1) * a + gate_weights[:, 1].unsqueeze(-1) * b
        
#         # Compute final attention score with residual connection and scaling
#         A = self.attention_c(A)
#         A = A / self.temperature  # Scaling
#         A = A + x.mean(dim=-1, keepdim=True)  # Residual connection for stability
        
#         return A, x
    

class Attn_Net_Gated(nn.Module):
    def __init__(self, L=1024, D=256, dropout=0.25, n_classes=1):
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = nn.Sequential(
            nn.Linear(L, D), # L is the visual_projection_dim
            nn.Tanh()
        )
        
        self.attention_b = nn.Sequential(
            nn.Linear(L, D),
            nn.Sigmoid()
        )
        
        if dropout:
            self.attention_a.add_module("Dropout", nn.Dropout(dropout))
            self.attention_b.add_module("Dropout", nn.Dropout(dropout))

        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a * b  # Element-wise multiplication for gated attention
        A = self.attention_c(A)
        return A, x

class Attn_Net(nn.Module):
    def __init__(self, L=1024, D=256, dropout=0.25, n_classes=1):
        super(Attn_Net, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(L, D),
            nn.Tanh(),  # Just a single non-linearity for attention scores
        )
        
        if dropout:
            self.attention.add_module("Dropout", nn.Dropout(dropout))

        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        A = self.attention(x)  # Single attention branch
        A = self.attention_c(A)
        return A, x


class CLAM_ViT(nn.Module):
    def __init__(self, base_model_name, training_config, gate=True, size_arg="small", dropout=0.25, k_sample=8,
                 device="cuda",
                 freeze_vit=False, freeze_query_features_encoder=False, use_cell_type_embedding=False):
        super(CLAM_ViT, self).__init__()

        # Load pre-trained ViT for patch feature extraction
        config = AutoConfig.from_pretrained(base_model_name)
        self.patch_encoder = AutoModel.from_pretrained(base_model_name, config=config)
        self.patch_processor = AutoProcessor.from_pretrained(base_model_name)
        self.freeze_vit = freeze_vit
        self.config = training_config
        self.freeze_query_features_encoder = freeze_query_features_encoder
        self.use_cell_type_embedding = use_cell_type_embedding
        self.device = device
        self.patch_encoder.to(self.device)
        
        # Unfreeze the patch encoder (ViT)
        if self.freeze_vit:
            for param in self.patch_encoder.vision_model.parameters():
                param.requires_grad = False
        else:
            for param in self.patch_encoder.vision_model.parameters():
                param.requires_grad = True
                
        if self.freeze_query_features_encoder:
            for param in self.patch_encoder.text_model.parameters():
                param.requires_grad = False
        else:
            for param in self.patch_encoder.text_model.parameters():
                param.requires_grad = True
        
        self.visual_projection_dim = self.patch_encoder.vision_model.encoder.layers[-1].mlp.fc2.out_features
        self.text_projection_dim = self.patch_encoder.text_model.encoder.layers[-1].mlp.fc2.out_features
        size_dict = {"tiny": [self.visual_projection_dim, 128],
                     "small": [self.visual_projection_dim, 256],
                     "big": [self.visual_projection_dim, 384]}
        size = size_dict[size_arg]
        
        if self.use_cell_type_embedding:
            # Replace the Embedding layer with a Linear layer
            self.cell_type_projection = nn.Linear(39, self.visual_projection_dim) #
            self.cell_type_projection.to(self.device)
            # self.wsi_vision_final_projection = nn.Linear(patch_dim, size[1])
            # self.wsi_vision_final_projection.to(self.device)
        
        # Attention network (gated or non-gated)
        if gate:
            self.attention_net = Attn_Net_Gated(L=size[0], D=size[1], dropout=dropout, n_classes=1)
        else:
            self.attention_net = Attn_Net(L=size[0], D=size[1], dropout=dropout, n_classes=1)

        self.k_sample = k_sample

        # Replace average pooling with a linear layer
        self.visual_token_projection = nn.Linear(self.visual_projection_dim, self.visual_projection_dim)

        # Add a projection layer from 512 to 768
        self.text_projection_to_visual_dim = nn.Linear(self.text_projection_dim, self.visual_projection_dim)

        # Task-specific heads: staining intensity, location, quantity
        self.classifiers = nn.ModuleList([
            nn.Linear(size[0], 4), # staining intensity
            nn.Linear(size[0], 4), # staining location
            nn.Linear(size[0], 4), # staining quantity
            nn.Linear(size[0], 58), # tissue type
            nn.Linear(size[0], 2) # tumor vs non-tumor
        ])

        # Add L2 regularization
        self.l2_reg = 1e-5

    def forward(self, patches, query_input, cell_type_one_hot, phase="test"):

        # patch_features_list = []
        # patch_counts = []
        # for ix, patch_tensor in enumerate(patches):
        #     # Extract patch features using ViT
        #     patch_outputs = self.patch_encoder.vision_model(patch_tensor, output_hidden_states=True)
        #     patch_tokens = patch_outputs.hidden_states[-1]  # Accessing the last layer's hidden states
        #     patch_features = self.visual_token_projection(patch_tokens)
        #     patch_features_list.append(patch_features)
        #     patch_counts.append(patch_features.shape[0])
        # # patch_features_list: [torch.Size([115, 50, 768]), torch.Size([111, 50, 768]), ..., torch.Size([104, 50, 768])] # N patches x N tokens x feature dim

        # Process patches in smaller chunks to save memory
        chunk_size = 32  # Adjust this value based on your GPU memory
        patch_features_list = []
        patch_counts = []
        
        for ix, patch_tensor in enumerate(patches):
            # Process patches in chunks
            num_patches = patch_tensor.size(0)
            chunk_features = []
            
            for i in range(0, num_patches, chunk_size):
                chunk = patch_tensor[i:i + chunk_size]
                with torch.cuda.amp.autocast():  # Use mixed precision
                    patch_outputs = self.patch_encoder.vision_model(chunk, output_hidden_states=True)
                    patch_tokens = patch_outputs.hidden_states[-1]
                    chunk_features.append(self.visual_token_projection(patch_tokens))
            
            # Concatenate chunks
            patch_features = torch.cat(chunk_features, dim=0)
            patch_features_list.append(patch_features)
            patch_counts.append(patch_features.shape[0])


        # Attention mechanism to weigh patch features
        A_list = []
        h_list = []
        for features in patch_features_list:
            # features: [N patches x N tokens x feature dim], for example, [115, 50, 768]
            A, h = self.attention_net(features) # here `h` is the same as `features`
            A_list.append(A)
            h_list.append(h)
        # A_list: [torch.Size([115, 50, 1]), torch.Size([111, 50, 1]), ..., torch.Size([104, 50,1])]
        # h_list: [torch.Size([115, 50, 768]), torch.Size([111, 50, 768]), ..., torch.Size([104, 50, 768])]

        # Process each sample separately
        M_list = []
        A_raw_list = []
        for A, h_value, count in zip(A_list, h_list, patch_counts):
            A = A[:count]  # Only keep attention weights for real patches
            A_raw = A.clone()
            A = F.softmax(A, dim=0)  # Softmax over patches for this sample # (115, 50, 1)
            # M = torch.mm(A.t(), h[:count])  # Weighted sum of features (1, 115) x (115, 512)
            M = (A * h_value).sum(dim=0) # [50, 768]            
            M_list.append(M)
            A_raw_list.append(A_raw)
            
        # Stack results
        A_raw = torch.cat(A_raw_list, dim=0)
        M = torch.stack(M_list) # 16 x 50 x 768

        M_single_token = M[:, 0, :] # CLS token 16 x 768
        M_single_token = torch.mean(M, dim=1) # Mean of all tokens 16 x 768
        

        if phase == "train":
            # Introduce a random ratio to toggle the addition of query_features
            if random.random() < 0.5:  # 50% chance to add query_features
                text_inputs = self.patch_processor.tokenizer(query_input, return_tensors="pt", padding=True, truncation=True, max_length=80)
                text_inputs.to(self.device)
                query_features = self.patch_encoder.get_text_features(**text_inputs) # N x 512
                query_features = self.text_projection_to_visual_dim(query_features)

                M_single_token = M_single_token + query_features
        
        if self.use_cell_type_embedding:
            cell_type_one_hot = cell_type_one_hot.to(self.device)
            # Use the linear projection instead of embedding
            cell_type_embed = self.cell_type_projection(cell_type_one_hot)
            # Expand cell_type_embed to match the batch size and append it to each patch feature
            cell_type_embed = cell_type_embed[ix].unsqueeze(0).expand(M.size(0), -1)
            M_single_token = M_single_token + cell_type_embed
        else:
            pass

        # Task-specific outputs
        outputs = [classifier(M_single_token) for classifier in self.classifiers]
        
        """
        Just use raw outputs, do not apply sigmoid or softmax!
        Because we will use CrossEntropy Loss, which already contains sigmoid
        If you apply sigmoid here, it will be a double sigmoid, and will destroy the gradient flow.
        Since the loss function expects raw logits and not sigmoid outputs, the calculated loss will be incorrect, and this will negatively impact the learning process.
        """
        intensity_out = outputs[0]
        location_out = outputs[1]
        quantity_out = outputs[2]
        tissue_out = outputs[3]
        malignancy_out = outputs[4]

        return intensity_out, location_out, quantity_out, tissue_out, malignancy_out, A_raw

    def print_last_layer_weights(self):
        print("Vision model last layer weights:")
        vision_last_layer = self.patch_encoder.vision_model.encoder.layers[-1].mlp.fc2
        print(vision_last_layer.weight.data)
        
        print("\nText model last layer weights:")
        text_last_layer = self.patch_encoder.text_model.encoder.layers[-1].mlp.fc2
        print(text_last_layer.weight.data)

    def get_l2_regularization(self):
        l2_loss = 0
        for param in self.parameters():
            l2_loss += torch.norm(param, 2)
        return self.l2_reg * l2_loss

    def get_model_state(self):
        return {
            'attention_net': self.attention_net.state_dict(),
            'wsi_vision_final_projection': self.wsi_vision_final_projection.state_dict() if hasattr(self, 'wsi_vision_final_projection') else None,
            'patch_encoder': self.patch_encoder.state_dict(),
            'classifiers': [clf.state_dict() for clf in self.classifiers]
        }
    
# Add a function to create an optimizer with learning rate scheduler
def create_optimizer_and_scheduler(model, lr=1e-4, weight_decay=1e-5):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=500, verbose=True)
    return optimizer, scheduler

def create_clam_vit(base_model_name, training_config, gate=True, size_arg="small", dropout=0.25, k_sample=8, n_classes=4, device="cuda",
                    lr=1e-4, weight_decay=1e-5,
                    freeze_vit=False, freeze_query_features_encoder=False, use_cell_type_embedding=False):
    model = CLAM_ViT(base_model_name=base_model_name, training_config=training_config, gate=gate, size_arg=size_arg, dropout=dropout,
                     k_sample=k_sample, device=device,
                     freeze_vit=freeze_vit, freeze_query_features_encoder=freeze_query_features_encoder, use_cell_type_embedding=use_cell_type_embedding)
    optimizer, scheduler = create_optimizer_and_scheduler(model, lr=lr, weight_decay=weight_decay)
    return model, optimizer, scheduler
