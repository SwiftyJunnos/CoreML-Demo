
import coremltools as ct
import torch
import torchvision

from coremltools.converters.mil.mil.program import Program
from coremltools.models.model import MLModel

## CIFAR

from cifar_classes import CIFAR_CLASSES
from net import Net

def convert_cifar(
    model_path: str,
    batch_size: int,
    device: torch.device
):
    model = Net()
    model.to(device)
    model.load_state_dict(torch.load(model_path)) # https://tutorials.pytorch.kr/beginner/saving_loading_models.html#state-dict
    # Inference O Training X
    model.eval() # Evaluation(평가) 모드로 변경 (미설정 시, 일관성 없는 결과)

    example_input = torch.zeros(1, 3, 32, 32, device=device) # shape 제공용
    # PyTorch가 아닌 CoreMLTools에서 사용하기 위한 변환
    traced_model = torch.jit.trace(model, example_input) # TorchScript (Torch 모델의 중간 표현(IR))

    # Image preprocessing
    scale = 1 / (0.5 * 255.0)

    # Classification Label 제공 (Output에 적용)
    classifier_config = ct.ClassifierConfig(class_labels=CIFAR_CLASSES)

    ml_model = ct.convert(
        traced_model,
        inputs=[ct.ImageType(name="image",
                             shape=example_input.shape,
                             bias=[-1.0, -1.0, -1.0])], # batch_size x 3 x 32 x 32
        outputs=[ct.TensorType(name="predictions")], # batch_size x 10
        # compute_precision=ct.precision.FLOAT16, #Typed Execution, default=FLOAT16
        minimum_deployment_target=ct.target.iOS15,
        classifier_config=classifier_config
    )

    ml_model.save("./model/CifarMLPackage.mlpackage")

## MobileNet

from urllib.request import urlopen
from PIL import Image
import timm

def convert_mobilenet():

    model = timm.create_model(
        'mobilenetv3_large_100.ra_in1k',
        pretrained=True
    )
    model.eval()
    example_input = torch.rand(1, 3, 224, 224)
    traced_model = torch.jit.trace(model, example_input)

    classifier_config = ct.ClassifierConfig(class_labels="converter/classes")

    ml_model = ct.convert(
        traced_model,
        inputs=[ct.ImageType(name="image", shape=example_input.shape)],
        outputs=[ct.TensorType(name="predictions")],
        minimum_deployment_target=ct.target.iOS15,
        classifier_config=classifier_config
    )

    ml_model.save("./model/MobileNetMLPackage.mlpackage")
