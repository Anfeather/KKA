from .flowers import Flowers_Dataset
from .ucm_caption import UCM_Caption_Dataset
from .mycifar import mycifar_Dataset


def load_dataset(dataset_name, data_path, normal_class, known_outlier_class, n_known_outlier_classes: int = 0,
                 ratio_known_normal: float = 0.0, ratio_known_outlier: float = 0.0, ratio_pollution: float = 0.0,
                 random_state=None):
    """Loads the dataset."""

    implemented_datasets = ('flower','ucm_caption','mycifar')
    assert dataset_name in implemented_datasets

    dataset = None

    if dataset_name == 'flower':
        dataset = Flowers_Dataset(
            root=data_path,
            normal_class=normal_class,
            known_outlier_class=known_outlier_class,
            n_known_outlier_classes=n_known_outlier_classes,
            ratio_known_normal=ratio_known_normal,
            ratio_known_outlier=ratio_known_outlier,
            ratio_pollution=ratio_pollution)
        

    if dataset_name == 'ucm_caption':  
        dataset = UCM_Caption_Dataset(
            root=data_path,
            normal_class=normal_class,
            known_outlier_class=known_outlier_class,
            n_known_outlier_classes=n_known_outlier_classes,
            ratio_known_normal=ratio_known_normal,
            ratio_known_outlier=ratio_known_outlier,
            ratio_pollution=ratio_pollution)


    if dataset_name == 'mycifar':  
        dataset = mycifar_Dataset(
            root=data_path,
            normal_class=normal_class,
            known_outlier_class=known_outlier_class,
            n_known_outlier_classes=n_known_outlier_classes,
            ratio_known_normal=ratio_known_normal,
            ratio_known_outlier=ratio_known_outlier,
            ratio_pollution=ratio_pollution)
        
    return dataset
