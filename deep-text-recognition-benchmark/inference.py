import time
from datetime import timedelta

import numpy as np
import timm
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torchvision.transforms as transforms
from dataset import AlignCollate, RawDataset
from PIL import Image
from text_recognition_model import TextRecognitionModel
from utils import AttnLabelConverter


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


# load text recognition model
def load_text_recognition_model(
    img_scale: tuple = (64, 200),  # (height, width)
    num_fiducial: int = 20,
    input_channel: int = 1,
    output_channel: int = 512,
    hidden_size: int = 256,
    character: str = "0123456789가강거경계고관광구금기김나남너노누다대더도동두등라러로루리마머명모무문미바배뱌버보부북사산서소수아악안양어연영오용우울원육이인자작저전조주중지차천초추충카타파평포하허호홀히",
    batch_max_length: int = 25,
    Transformation: str = "TPS",
    FeatureExtraction: str = "VGG",
    SequenceModeling: str = "BiLSTM",
    Prediction: str = "Attn",
    save_model: str = None,
    device: str = "cuda",
):
    """
    text recognition model 을 불러오는 함수

    Args:
        save_model(str) : 모델 weight 가 저장된 pth 파일 경로
        device(str) : cuda or cpu
    """

    assert save_model is not None
    # assert Transformation == "TPS" and FeatureExtraction == "VGG"
    assert SequenceModeling == "BiLSTM" and Prediction == "Attn"

    converter = AttnLabelConverter(character)
    num_class = len(converter.character)

    print(f"loading pretrained model from {save_model}")

    model = TextRecognitionModel(
        Transformation,
        FeatureExtraction,
        SequenceModeling,
        Prediction,
        num_fiducial,
        img_scale,
        input_channel,
        output_channel,
        hidden_size,
        num_class,
        batch_max_length,
    )
    model = torch.nn.DataParallel(model).to(device)
    # load model
    model.load_state_dict(torch.load(save_model, map_location=device))
    print("load model..")

    return model


class ResizeNormalize(object):
    def __init__(self, size, interpolation=Image.BICUBIC):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()

    def __call__(self, img):
        img = img.resize(self.size, self.interpolation)
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        return img


def inference(
    model=None,
    input_tensor: torch.Tensor = None,
    batch_max_length: int = 25,
    character: str = "0123456789가강거경계고관광구금기김나남너노누다대더도동두등라러로루리마머명모무문미바배뱌버보부북사산서소수아악안양어연영오용우울원육이인자작저전조주중지차천초추충카타파평포하허호홀히",
    device: str = "cuda",
):
    """
    text recognition model로 단일 이미지에 대한 결과를 출력하는 함수

    Args:
        model: load_ocr_model
        img_array : img
        device(str): cpu or cuda
    """

    converter = AttnLabelConverter(character)
    model.eval()

    # predict
    with torch.inference_mode():
        if len(input_tensor.shape) == 3:
            input_tensor = input_tensor.unsqueeze(0)

        # for image_tensors, image_path_list in ocr_data_loader:
        batch_size = input_tensor.size(0)
        image = input_tensor.to(device)

        # For max length prediction
        length_for_pred = torch.IntTensor([batch_max_length] * batch_size).to(device)
        text_for_pred = torch.LongTensor(batch_size, batch_max_length + 1).fill_(0).to(device)

        start = time.time()
        pred = model(image, text_for_pred, is_train=False)
        print(f"ocr: {timedelta(seconds=time.time() - start)}")

        # select max probabilty (greedy decoding) then decode index to character
        _, pred_index = pred.max(2)
        pred_str = converter.decode(pred_index, length_for_pred)
        pred_prob = F.softmax(pred, dim=2)
        pred_max_prob, _ = pred_prob.max(dim=2)

        preds = []
        confidence_scores = []
        for pred, pred_max_prob in zip(pred_str, pred_max_prob):
            pred_EOS = pred.find("[s]")
            pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
            pred_max_prob = pred_max_prob[:pred_EOS]

            # calculate confidence score (= multiply of pred_max_prob)
            confidence_score = pred_max_prob.cumprod(dim=0)[-1]
            preds.append(pred)
            confidence_scores.append(confidence_score.item())

        return preds, confidence_scores


if __name__ == "__main__":
    # load model
    model = load_text_recognition_model(save_model="best_accuracy.pth", device="cuda")
    # inference
    sample_img = Image.open("sample_img").convert("L")
    result, confidence_score = inference(model=model, img_array=sample_img, device="cuda")
    print(f"{result:25s}\t{confidence_score:0.4f}")
