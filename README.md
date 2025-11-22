moved output/, data/, pretrained_model/ to outside of MVT/

`det-mvt/`  
* `data/`  
    * contains depthtrack dataset
    * download depthtrack data with `scripts/download_depthtrack.sh`
* `output/`
    * checkpoints, pretrained networks, tensorboard
    * download pretrained mobilevit_s.pt with `scripts/download_pretrained_mobilevit.sh`
* `DeT/`
    * run experiments with `scripts/train*.sh` and `scripts/test*.sh`