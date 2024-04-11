
from PIL import Image
from transformers import ViTImageProcessor, ViTForImageClassification

import coremltools as ct
import coremltools.optimize.coreml as cto
import datasets
import torch
import torch.nn as nn
import torchvision.transforms as transforms

from imagenet_classes import IMAGENET2012_CLASSES

__VERSION__: str = "1.0.2"

MODEL_NAME = "google/vit-base-patch16-224"
SAMPLE_IMAGE = "./image.jpg"
MODEL_OUTPUT_PATH = "../model/ViTMLPackage.mlpackage"
COMPILED_OUTPUT_PATH = "../model/ViTMLPackage.mlmodelc"

class_values = [val for key, val in IMAGENET2012_CLASSES.items()]

class ViTLogitsModel(torch.nn.Module):
    def __init__(self, model_name):
        super(ViTLogitsModel, self).__init__()
        # Load the pre-trained model
        self.vit_model = ViTForImageClassification.from_pretrained(model_name)
        self.softmax = nn.Softmax()

    def forward(self, input_ids):
        # Get only model outputs' logits
        outputs = self.vit_model(input_ids).logits
        outputs = self.softmax(outputs)
        return outputs

def train_vit():
    model = ViTLogitsModel(MODEL_NAME)
    model.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # (평균, 분산)
    ])

    example_input = torch.rand(1, 3, 224, 224)
    traced_model = torch.jit.trace(model, example_input)

    classifier_config = ct.ClassifierConfig(class_labels=class_values)

    ml_model = ct.convert(
        traced_model,
        inputs=[ct.ImageType(
            name="image",
            shape=example_input.shape,
            bias=[-1.0, -1.0, -1.0],
            scale=(1 / (0.5 * 255.0)),
            color_layout=ct.colorlayout.RGB
        )],
        outputs=[ct.TensorType(name="prediction")],
        minimum_deployment_target=ct.target.iOS15,
        # classifier_config=classifier_config # coremltools 🐞 Bug: https://github.com/apple/coremltools/issues/1906
    )

    ml_model.short_description = "Based on Google's vit-base-patch16: https://huggingface.co/google/vit-base-patch16-224"
    ml_model.license = "apache-2.0"
    ml_model.version = __VERSION__

    ## Optimizing

    # Pruning
    # 영향도가 낮은 Weight를 0으로 만들고, 해당 값들을 제거 (index만 별도 보관)
    config = cto.OptimizationConfig(
        global_config=cto.OpThresholdPrunerConfig(threshold=1e-12)
    )
    compressed_model = cto.prune_weights(ml_model, config)

    # Quantization
    # Scale을 조정하고 Bias를 더해주는 것으로 값의 범위를 축소 (ex: Float16 -> Int8)
    config = cto.OptimizationConfig(
        global_config=cto.OpLinearQuantizerConfig(mode="linear_symmetric")
    )
    compressed_model = cto.linear_quantize_weights(compressed_model, config)

    # Palettization
    # Weight를 몇 가지 값들로 팔레트 화 (ex: 6.9, 6.3 -> 6.5)
    # LookUp 테이블을 만들어 Key 값을 보관 (4-bit: 0000 ~ 1111 -> 16개의 값 사용)
    config = cto.OptimizationConfig(
        global_config=cto.OpPalettizerConfig(mode="kmeans", nbits=4)
    )
    compressed_model = cto.palettize_weights(compressed_model, config)

    # 173.3MB -> 87.2MB
    # 사용할 때는 165MB 정도..
    compressed_model.save(MODEL_OUTPUT_PATH)

def test_vit(classes: list[str]):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # (평균, 분산)
    ])

    model = ct.models.MLModel(MODEL_OUTPUT_PATH)

    # Weight (type), NN 구조 조회
    # print(cto.get_weights_metadata(model))

    # Prepare model & image (input)
    model = ct.models.CompiledMLModel(COMPILED_OUTPUT_PATH)
    image = Image.open(SAMPLE_IMAGE).resize((224, 224))

    outputs: torch.Tensor = model.predict({"image": image}) # data must be a dict
    # print(outputs)
    predicted_class_idx = int(outputs["prediction"].argmax(-1).item())
    print("Predicted class:", classes[predicted_class_idx])

train_vit()
test_vit(classes=class_values)
