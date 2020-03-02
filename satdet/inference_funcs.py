from models import build_detector
from tools import inference_detector
from utils import load_checkpoint, Config


cfg = Config.fromfile('/satdetection/satdet/configs/faster_rcnn_x101_64x4d_fpn_1x_dota.py')
cfg.model.pretrained = None

model = build_detector(cfg.model, test_cfg=cfg.test_cfg)
_ = load_checkpoint(model, '/satdetection/satdet/trained_checkpoints/epoch1_8078_finetune0002.pth')

print('------The model has been loaded.-----')


def inference_single_func(img_path):
    result = inference_detector(model, img_path, cfg, device='cuda:0')
    #print("Array length: ", len(result))
    #for i in range(0, len(result)):
    #    print("[", i, "]: ", result[i])
    return result
