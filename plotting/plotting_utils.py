from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt
import os

TENSORBOARD_DIR = '/mnt/det-mvt/outputs/tensorboard'

def load_from_tensorboard(model_dir='dimp/DeT_DiMP50_Max', default_name=''):
    log_dir_train = os.path.join(TENSORBOARD_DIR, "ltr", model_dir, "train")
    log_dir_test = os.path.join(TENSORBOARD_DIR, "ltr", model_dir, "val")
    try:
        ea_train = event_accumulator.EventAccumulator(log_dir_train)
        ea_train.Reload()
        ea_test = event_accumulator.EventAccumulator(log_dir_test)
        ea_test.Reload()
        print(f'Loaded {model_dir}')
        # print(ea_test.Tags()["scalars"])
    except:
        print(f'{model_dir} does not exist, skipping')
        return {'default_name': default_name}

    # Convert events to lists
    try:
        model_train_epochs = [e.step for e in ea_train.Scalars("Loss/total")]
        model_train_loss = [e.value for e in ea_train.Scalars("Loss/total")]
        model_train_iou = [e.value for e in ea_train.Scalars("Loss/iou")]
        model_train_clf = [e.value for e in ea_train.Scalars("Loss/target_clf")]
    except:
        print('warning: train data does not exist')
        model_train_epochs = []
        model_train_loss = []
        model_train_iou = []
        model_train_clf = []

    try:
        model_val_epochs = [e.step for e in ea_test.Scalars("Loss/total")]
        model_val_loss = [e.value for e in ea_test.Scalars("Loss/total")]
        model_val_iou = [e.value for e in ea_test.Scalars("Loss/iou")]
        model_val_clf = [e.value for e in ea_test.Scalars("Loss/target_clf")]
    except:
        print('warning: val data does not exist')
        model_val_epochs = []
        model_val_loss = []
        model_val_iou = []
        model_val_clf = []

    return {'Epochs': {'train': model_train_epochs, 
                       'val': model_val_epochs},
            'Loss': {'train': model_train_loss, 
                     'val': model_val_loss},
            'IoU Loss': {'train': model_train_iou, 
                     'val': model_val_iou},
            'Clf Loss': {'train': model_train_clf, 
                     'val': model_val_clf},
            'default_name': default_name,
            }

COLORS = {
            # 'Loss': {'train': '#7fc9ff', 
            #             'val': '#0073e6'},
            # 'F1': {'train': '#ffa9a9', 
            #             'val': '#e60000'},
            'IoU Loss': {'train': '#ffa9a9', 
                        'val': '#e60000'},
            'Clf Loss': {'train': '#7fc9ff', 
                        'val': '#0073e6'},
            }


FONT_SIZE = 12

def plot(axes, i, model_metrics, metric_to_plot: str, model_name='', y_range=None):
    start_idx = 1 # 0 or 1
    if 'Epochs' in model_metrics:
        axes[i].plot(
            model_metrics['Epochs']['train'][start_idx:],
            model_metrics[metric_to_plot]['train'][start_idx:],
            marker='o',
            color=COLORS[metric_to_plot]['train'],
            label='Train'
        )
        axes[i].plot(
            model_metrics['Epochs']['val'],
            model_metrics[metric_to_plot]['val'],
            marker='o',
            color=COLORS[metric_to_plot]['val'],
            label='Val'
        )

    if model_name == '':
        model_name = model_metrics['default_name']

    axes[i].set_xlabel("Epochs", fontsize=FONT_SIZE)
    axes[i].set_ylabel("Loss", fontsize=FONT_SIZE)

    axes[i].set_title(model_name + ' ' + metric_to_plot, fontsize=FONT_SIZE)
    axes[i].legend(fontsize=FONT_SIZE)
    axes[i].grid(True)

    if y_range is not None:
        axes[i].set_ylim(y_range)


def plot_all(model_metrics_list, metric_to_plot, y_range=None):
    num_plots = len(model_metrics_list)
    fig, axes = _subplot_maker(num_plots)

    for i in range(num_plots):
        plot(axes, i, model_metrics_list[i], metric_to_plot, y_range=y_range)
     
        
def _subplot_maker(num_plots):
    if num_plots == 2:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    elif num_plots == 3:
        fig, axes = plt.subplots(1, 3, figsize=(13.5, 4))
    elif num_plots == 4:
        fig, axes = plt.subplots(1, 4, figsize=(15, 3))
    elif num_plots == 5:
        fig, axes = plt.subplots(1, 5, figsize=(16, 2.5))
    elif num_plots == 6:
        fig, axes = plt.subplots(1, 6, figsize=(16, 2.5))
    else:
        assert False
    
    return fig, axes

def save_plot(save_path=''):
    plt.tight_layout()
    if save_path is not '':
        plt.savefig(save_path)
