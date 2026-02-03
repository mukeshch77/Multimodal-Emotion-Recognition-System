
import torch
import torch.nn as nn
from transformers import BertModel
from torchvision import models

class BERT_ResNet_Fusion(nn.Module):
    def __init__(self, num_classes):
        super(BERT_ResNet_Fusion, self).__init__()
        
        # ----- TEXT ENCODER -----
        self.text_model = BertModel.from_pretrained('bert-base-uncased')

        # ----- IMAGE ENCODER -----
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad = False
        self.image_model = nn.Sequential(*list(resnet.children())[:-1])  # remove FC layer

        # ----- FUSION LAYER -----
        self.fc_fusion = nn.Sequential(
            nn.Linear(768 + 2048, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, input_ids, attention_mask, images):
        text_out = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        text_feat = text_out.pooler_output  # (batch, 768)

        img_feat = self.image_model(images)
        img_feat = img_feat.view(img_feat.size(0), -1)  # (batch, 2048)

        fused = torch.cat((text_feat, img_feat), dim=1)
        logits = self.fc_fusion(fused)
        return logits
