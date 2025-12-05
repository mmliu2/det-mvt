from plotting_utils import load_from_tensorboard, plot, plot_all, save_plot
import matplotlib.pyplot as plt


# load merics
dimp = load_from_tensorboard('dimp/DeT_DiMP50_Max', default_name='DeT-DiMP50')
mvt = load_from_tensorboard('mvt/DeT_MVT_Max', default_name='DeT-MVT')

mvt_rs_kd1 = load_from_tensorboard('mvt/DeT_MVT_Max_KD001', default_name='DeT-MVT+rsKD')
mvt_rs_kd2 = load_from_tensorboard('mvt/DeT_MVT_Max_KD002', default_name='DeT-MVT+2rsKD')
mvt_rs_crd1 = load_from_tensorboard('mvt/DeT_MVT_Max_CRD001', default_name='DeT-MVT+rsCRD')
mvt_rs_crd2 = load_from_tensorboard('mvt/DeT_MVT_Max_CRD002', default_name='DeT-MVT+2rsCRD')
mvt_rs_kd1_crd1 = load_from_tensorboard('mvt/DeT_MVT_Max_KD001_CRD001', default_name='DeT-MVT+rsKD+rsCRD')

mvt_cf_kd1 = load_from_tensorboard('mvt/DeT_MVT_Max_cf_KD001', default_name='DeT-MVT+cfKD')
mvt_cf_kd2 = load_from_tensorboard('mvt/DeT_MVT_Max_cf_KD002', default_name='DeT-MVT+2cfKD')
mvt_cf_crd1 = load_from_tensorboard('mvt/DeT_MVT_Max_cf_CRD001', default_name='DeT-MVT+cfCRD')
mvt_cf_crd2 = load_from_tensorboard('mvt/DeT_MVT_Max_cf_CRD002', default_name='DeT-MVT+2cfCRD')
mvt_cf_kd1_crd1 = load_from_tensorboard('mvt/DeT_MVT_Max_cf_KD001_CRD001', default_name='DeT-MVT+cfKD+cfCRD')

mvt_cf_kd4_unused = load_from_tensorboard('mvt/DeT_MVT_Max_cf_KD004_unused', default_name='DeT-MVT+4rsKD(unused)')


metrics = [mvt, mvt_rs_kd1, mvt_rs_kd2, mvt_rs_crd1, mvt_rs_crd2, mvt_rs_kd1_crd1]
plot_all(metrics, 'IoU Loss', y_range=(0.08, 0.25))
save_plot('/mnt/det-mvt/plotting/figs/loss/rs_iou.png')
plot_all(metrics, 'Clf Loss', y_range=(0.225, 0.55)) # 0.425
save_plot('/mnt/det-mvt/plotting/figs/loss/rs_clf.png')

metrics = [mvt, mvt_cf_kd1, mvt_cf_kd2, mvt_cf_crd1, mvt_cf_crd2, mvt_cf_kd1_crd1]
plot_all(metrics, 'IoU Loss', y_range=(0.08, 0.25))
save_plot('/mnt/det-mvt/plotting/figs/loss/cf_iou.png')
plot_all(metrics, 'Clf Loss', y_range=(0.225, 0.55))
save_plot('/mnt/det-mvt/plotting/figs/loss/cf_clf.png')