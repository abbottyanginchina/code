import mmengine
from vti_utils.utils import get_all_datasets


original_data = get_all_datasets(cfg)

if __name__ == '__main__':
    config_path = f'configs/cfgs.yaml'
    cfg = mmengine.Config.fromfile(config_path)