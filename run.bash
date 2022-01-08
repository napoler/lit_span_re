# shellcheck disable=SC1113
# /bin/bash

python bulidDataset_baidukg.py
python trainer.py fit --config config/config_dev.yaml
