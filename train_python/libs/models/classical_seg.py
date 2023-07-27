import torch
import torch.nn as nn
import torchvision.models as models


class Deeplab(nn.Module):
    def __init__(self, numClasses=3, weights='COCO_WITH_VOC_LABELS_V1'):
        super(Deeplab, self).__init__()

        deeplab = models.segmentation.deeplabv3_mobilenet_v3_large(weights=weights,
                                                         progress=1,
                                                         aux_loss=True)
        
        # print(deeplab)
        deeplab.classifier[4] = nn.Conv2d(256,
                                          numClasses,
                                          kernel_size=(1, 1),
                                          stride=(1, 1))

        deeplab.aux_classifier[4] = nn.Conv2d(10,
                                              numClasses,
                                              kernel_size=(1, 1),
                                              stride=(1, 1))

        self.dl = deeplab

        self.actfunction = nn.Softmax(dim=1)

    def forward(self, x):

        output = self.dl(x)['out']
        output = self.actfunction(output)
        return output


class LRASPP(nn.Module):
    def __init__(self, numClasses=3, weights='COCO_WITH_VOC_LABELS_V1'):
        super(LRASPP, self).__init__()

        lraspp = models.segmentation.lraspp_mobilenet_v3_large(weights=weights)

        lraspp.classifier.low_classifier = nn.Conv2d(40,
                                                     numClasses,
                                                     kernel_size=(1, 1),
                                                     stride=(1, 1))

        lraspp.classifier.high_classifier = nn.Conv2d(128,
                                                      numClasses,
                                                      kernel_size=(1, 1),
                                                      stride=(1, 1))
        self.lraspp = lraspp
        self.actfunction = nn.Softmax(dim=1)

    def forward(self, x):

        output = self.lraspp(x)['out']
        output = self.actfunction(output)
        return output


if __name__ == "__main__":
    input = torch.randn(64, 3, 224, 224)
    model = Deeplab()
    print(model)
    pred = model(input)
    print(pred.size())
