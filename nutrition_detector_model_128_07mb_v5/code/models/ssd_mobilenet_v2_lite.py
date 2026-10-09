import config as cfg
from models.ssd_mobilenet_v2 import build_ssd_model


def build_student(return_features=True, pretrained=True):
    return build_ssd_model(
        alpha=cfg.STUDENT_ALPHA,
        fpn_channels=cfg.STUDENT_FPN_CHANNELS,
        pretrained=pretrained,
        return_features=return_features,
        name="student_ssd",
    )
