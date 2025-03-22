import torch
import torch.nn as nn
from torchvision.models import inception_v3


class InceptionV3FeatureExtractor(nn.Module):
    def __init__(self) -> None:
        super(InceptionV3FeatureExtractor, self).__init__()
        # Load pretrained InceptionV3 with aux_logits disabled.
        inception = inception_v3(pretrained=True, transform_input=False, aux_logits=True)
        inception.eval()
        self.inception = inception

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # InceptionV3 expects 299x299 images.
        # Forward through the network until the final pooling layer to extract features.
        x = self.inception.Conv2d_1a_3x3(x)
        x = self.inception.Conv2d_2a_3x3(x)
        x = self.inception.Conv2d_2b_3x3(x)
        x = self.inception.maxpool1(x)
        x = self.inception.Conv2d_3b_1x1(x)
        x = self.inception.Conv2d_4a_3x3(x)
        x = self.inception.maxpool2(x)
        x = self.inception.Mixed_5b(x)
        x = self.inception.Mixed_5c(x)
        x = self.inception.Mixed_5d(x)
        x = self.inception.Mixed_6a(x)
        x = self.inception.Mixed_6b(x)
        x = self.inception.Mixed_6c(x)
        x = self.inception.Mixed_6d(x)
        x = self.inception.Mixed_6e(x)
        x = self.inception.Mixed_7a(x)
        x = self.inception.Mixed_7b(x)
        x = self.inception.Mixed_7c(x)
        # Final average pooling and flattening to get the feature vector.
        x = self.inception.avgpool(x)
        x = torch.flatten(x, 1)
        return x
