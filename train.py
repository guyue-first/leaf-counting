import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

# BILIBILI UP 魔傀面具
# 训练参数官方详解链接：https://docs.ultralytics.com/modes/train/#resuming-interrupted-trainings:~:text=a%20training%20run.-,Train%20Settings,-The%20training%20settings
from datetime import datetime
# 指定显卡和多卡训练问题 统一都在<YOLOV11配置文件.md>下方常见错误和解决方案。
# 训练过程中loss出现nan，可以尝试关闭AMP，就是把下方amp=False的注释去掉。
# 训练时候输出的AMP Check使用的YOLO11n的权重不是代表载入了预训练权重的意思，只是用于测试AMP，正常的不需要理会。
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# 定义基本路径
base_project_path = 'runs/train'

# 创建带有时间戳的文件夹名
exp_name = f'yolo11-SPPF-LSKA-SDFM-dyhead_{timestamp}'

# exp_name = f'update-yolo11-SPPF-LSKA-ContextGuideFPN_{timestamp}'

if __name__ == '__main__':
    # model = YOLO('ultralytics/cfg/models/11/yolo11n-WaveletPool.yaml')
    model = YOLO('ultralytics/cfg/models/11-update/yolo11n-SPPF-LSKA-SDFM-dyhead.yaml')
    model.load('yolo11n.pt')  # loading pretrain weights
    model.train(data='ultralytics/cfg/datasets/coco8.yaml',
                cache=False,
                imgsz=640,
                epochs=100,
                batch=16,
                close_mosaic=0,
                workers=4,  # Windows下出现莫名其妙卡主的情况可以尝试把workers设置为0
                # device='0',
                optimizer='SGD',  # using SGD
                # patience=0, # set 0 to close earlystop.
                # resume=True, # 断点续训,YOLO初始化时选择last.pt,例如YOLO('last.pt')
                # amp=False, # close amp
                # fraction=0.2,
                project='runs/train2',
                name=exp_name,
                )