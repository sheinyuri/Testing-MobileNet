import os
from pathlib import Path

mode = 0o755

# datasets = ['balanced', 'imbalanced']
# loss_functions = ['ce', 'label_smoothing','focal_loss',"class_balanced",'weighted_ce']
# seeds = [45, 2, 255, 18, 33]

parent_path = "/students/u7947738/Documents/comp3242/project_alt/Testing-MobileNet/training_data"


def make_folders(parent, data, losses, seeds):
    os.mkdir(parent_path)

    for dataset in datasets:
        dataset_folder = parent_path + "/" + dataset
        os.mkdir(dataset_folder,mode)
        for loss_function in loss_functions[:len(loss_functions)- (2 if dataset == "balanced" else 0)]:
            loss_folder = dataset_folder + "/" + loss_function
            os.mkdir(loss_folder,mode)
            for seed in seeds:
                seed_folder = loss_folder + "/seed_" + str(seed)
                os.mkdir(seed_folder,mode)
                with open(os.path.join(seed_folder, 'ignore.txt'),"w") as f:
                    pass