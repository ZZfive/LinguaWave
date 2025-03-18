# SDK模型下载
from modelscope import snapshot_download
# snapshot_download('iic/CosyVoice-300M-25Hz', local_dir='/home/dbt/zjy/CosyVoice/pretrained_models/v2/CosyVoice-300M-25Hz')
# snapshot_download('iic/CosyVoice-300M-SFT', local_dir='/home/dbt/zjy/CosyVoice/pretrained_models/v2/CosyVoice-300M-SFT')
# snapshot_download('iic/CosyVoice-300M-Instruct', local_dir='/home/dbt/zjy/CosyVoice/pretrained_models/v2/CosyVoice-300M-Instruct')
# snapshot_download('iic/CosyVoice-ttsfrd', local_dir='/home/dbt/zjy/CosyVoice/pretrained_models/v2/CosyVoice-ttsfrd')

snapshot_download('iic/CosyVoice2-0.5B', local_dir='/root/weight/CosyVoice2-0.5B')
# snapshot_download('iic/CosyVoice-ttsfrd', local_dir='/home/dbt/zjy/CosyVoice/pretrained_models/v2/CosyVoice-ttsfrd')