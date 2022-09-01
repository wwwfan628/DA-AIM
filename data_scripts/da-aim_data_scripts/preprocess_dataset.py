"""Creates the annotation files for a certain configuration given the original ava annotation and frame list files"""
"""This particular version leaves the classids as in the original file but still performs the reduction"""
import os
import shutil
import numpy as np
import pandas as pd
import argparse
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import json

writer = SummaryWriter()


def preprocess_ava_action_list(source_path, target_paths, CLASSES, NEW_LABEL_ID):
    name = None
    label_id = None
    labels = {}
    with open(source_path, "r") as source_file:
        for line in source_file:
            line.replace(" ", "")
            if line == "label {\n":
                pass
            elif line == "}\n":
                pass
            elif "name" in line:
                name = line.split(":", 1)[1].strip()
            elif "id" in line:
                label_id = int(line.split(":", 1)[1].strip())

            if name is not None and label_id is not None and label_id in CLASSES:
                # modify original id to new id
                new_label_id = NEW_LABEL_ID.__getitem__(label_id)
                labels[name] = new_label_id
                label_id = None
                name = None

    for target_path in target_paths:
        with open(target_path, "w") as target_file:
            for key, content in labels.items():
                target_file.write("label {\n")
                target_file.write('  name: {}\n'.format(key))
                target_file.write('  label_id: {}\n'.format(int(content)))
                target_file.write("}\n")


def preprocess_dataset_train(annotations_source_path, annotations_target_path, boxes_source_path, boxes_target_path,
                             frame_list_source_path, frame_list_target_path, CLASSES, num_samples_per_class,
                             NEW_LABEL_ID, exclusive=False):
    classes = CLASSES.copy()
    # read csv file
    annotations_train = pd.read_csv(annotations_source_path, header=None)
    annotations_train_all = annotations_train     # save copy of original df, because will be modified later
    if boxes_source_path != '':
        boxes_train = pd.read_csv(boxes_source_path, header=None)
        boxes_train_all = boxes_train       # save copy of original df, because will be modified later
    frame_list_train = pd.read_csv(frame_list_source_path, header=0, sep=' ')
    if num_samples_per_class != float('inf'):
        # find the target class from CLASSES that has smallest number of samples
        if exclusive:
            annotations_train = annotations_train[annotations_train[6].isin(classes)]
            mask = annotations_train.groupby([annotations_train[0], annotations_train[1], annotations_train[2],
                        annotations_train[3], annotations_train[4], annotations_train[5]])[0].transform('size') == 1
            annotations_train = annotations_train[mask]
            current_target_class = annotations_train[6].value_counts().keys().array[-1]
            class_counts_original = annotations_train[6].value_counts()
        else:
            annotations_train = annotations_train[annotations_train[6].isin(classes)]
            current_target_class = annotations_train[6].value_counts().keys().array[-1]
            class_counts_original = annotations_train[6].value_counts()
        # record the number of samples in reduced dataset
        class_counts = np.zeros(len(classes), dtype=int)
        # extract wanted samples from the original dataset
        annotations_train_list = []
        # record wanted samples: video_id + time_stamp
        samples_ids = []
        # number of total samples
        num_samples = 0
        # collect samples until reaching target number
        while True:
            annotations_train_current_target_class = annotations_train[annotations_train[6] == current_target_class]
            idx = np.random.randint(len(annotations_train_current_target_class))
            video_id = annotations_train_current_target_class.iloc[idx][0]
            time_stamp = annotations_train_current_target_class.iloc[idx][1]
            if video_id + ' ' + str(time_stamp) in samples_ids:
                continue
            else:
                frame_annotations = annotations_train[np.logical_and(annotations_train[1] == time_stamp, annotations_train[0] == video_id)]
                frame_has_qualified_bbox = False
                for index, annotation in frame_annotations.iterrows():
                    if annotation[6] in classes:
                        frame_has_qualified_bbox = True
                        annotations_train_list.append(annotations_train_all.iloc[[index]])
                        class_counts_idx = classes.index(annotation[6])
                        class_counts[class_counts_idx] = class_counts[class_counts_idx] + 1
                        num_samples += 1
                        print("Video {} | Time stamp {} | Class {}".format(video_id, time_stamp, annotation[6]))
                if frame_has_qualified_bbox:
                    samples_ids.append(video_id + ' ' + str(time_stamp))
                # check if any class has satisfy the request: 1. number of samples reached num_samples_per_class
                # or 2. all samples from original class has been collected
                to_delete_idx = []   # if satisfing request, delete from list
                for class_counts_idx, class_count in enumerate(class_counts):
                    if class_count >= num_samples_per_class or class_count == class_counts_original.loc[classes[class_counts_idx]]:
                        to_delete_idx.append(class_counts_idx)
                if len(to_delete_idx) != 0:
                    for idx in sorted(to_delete_idx, reverse=True):
                        print('Delete class {} from searching list. {} samples from this class has been collected.'.format(
                                classes[idx], class_counts[idx]))
                        del classes[idx]
                        class_counts = np.delete(class_counts, idx)
                # update current target class: class that contains smallest number of samples
                if len(classes) != 0:
                    current_target_class_idx = class_counts.argmin()
                    current_target_class = classes[current_target_class_idx]
                else:
                    break
        annotations_train = pd.concat(annotations_train_list)
        print('After reduction, training dataset contains {} samples.'.format(num_samples))
        print('After reduction, training dataset contains {} key frames.'.format(len(samples_ids)))
    else:
        # find all samples of wanted classes from CLASSES
        annotations_train = annotations_train[annotations_train[6].isin(classes)]
        if exclusive:
            mask = annotations_train.groupby([annotations_train[0], annotations_train[1], annotations_train[2],
                        annotations_train[3], annotations_train[4], annotations_train[5]])[0].transform('size') == 1
            annotations_train = annotations_train[mask]
        print('After reduction, training dataset contains {} samples.'.format(len(annotations_train)))
        print('After reduction, training dataset contains {} key frames.'.format(
            len(annotations_train.groupby([annotations_train[0], annotations_train[1]]))))
    # sort for the first two columns
    annotations_train = annotations_train.sort_values(by=[0, 1])
    # replace original id with new id
    annotations_train[6] = annotations_train[6].apply(lambda x: NEW_LABEL_ID.__getitem__(x))
    # save annotations
    annotations_train.to_csv(annotations_target_path, header=None, index=False, float_format='%.3f')

    # preprocess predicted boxes file
    if boxes_source_path != '':
        classes = CLASSES.copy()
        classes.append(-1)
        new_label_id = NEW_LABEL_ID.copy()
        new_label_id[-1] = -1
        if exclusive:
            boxes_train = boxes_train[boxes_train[6].isin(classes)]
            mask = boxes_train.groupby([boxes_train[0], boxes_train[1], boxes_train[2], boxes_train[3],
                                        boxes_train[4], boxes_train[5]])[0].transform('size') == 1
            boxes_train = boxes_train[mask]
        else:
            boxes_train = boxes_train[boxes_train[6].isin(classes)]
        samples_ids = annotations_train[[0, 1]].value_counts().keys()
        boxes_train_list = []
        for video_id, time_stamp in samples_ids:
            frame_boxes = boxes_train[np.logical_and(boxes_train[1] == time_stamp, boxes_train[0] == video_id)]
            for index, box in frame_boxes.iterrows():
                boxes_train_list.append(boxes_train_all.iloc[[index]])
                print("Video {} | Time stamp {} | Class {}".format(video_id, time_stamp, box[6]))
            boxes_train = pd.concat(boxes_train_list)
        # sort for the first two columns
        boxes_train = boxes_train.sort_values(by=[0, 1])
        # replace original id with new id
        boxes_train[6] = boxes_train[6].apply(lambda x: new_label_id.__getitem__(x))
        # save annotations
        boxes_train.to_csv(boxes_target_path, header=None, index=False, float_format='%.3f')

    # preprocess frame list
    frame_list_train = frame_list_train[frame_list_train['original_vido_id'].isin(annotations_train[0].unique())]
    frame_list_train = frame_list_train.replace(np.nan, -1)
    frame_list_train.to_csv(frame_list_target_path, sep=' ', index=False)


def preprocess_dataset_val(annotations_source_path, annotations_target_paths, boxes_source_path, boxes_target_paths,
                        frame_list_source_path, frame_list_target_paths, CLASSES, NEW_LABEL_ID, exclusive=False):
    # read csv file
    annotations_val = pd.read_csv(annotations_source_path, header=None)
    if boxes_source_path != '':
        boxes_val = pd.read_csv(boxes_source_path, header=None)
    frame_list_val = pd.read_csv(frame_list_source_path, header=0, sep=' ')
    # find all samples of wanted classes from CLASSES
    annotations_val = annotations_val[annotations_val[6].isin(CLASSES)]
    if exclusive:
        mask = annotations_val.groupby([annotations_val[0], annotations_val[1], annotations_val[2],
                        annotations_val[3], annotations_val[4], annotations_val[5]])[0].transform('size') == 1
        annotations_val = annotations_val[mask]
    print('After reduction, validation dataset contains {} samples.'.format(len(annotations_val)))
    print('After reduction, validation dataset contains {} key frames.'.format(len(annotations_val.groupby([annotations_val[0], annotations_val[1]]))))
    # sort for the first two columns
    annotations_val = annotations_val.sort_values(by=[0, 1])
    # replace original id with new id
    annotations_val[6] = annotations_val[6].apply(lambda x: NEW_LABEL_ID.__getitem__(x))
    for annotations_target_path in annotations_target_paths:
        annotations_val.to_csv(annotations_target_path, header=None, index=False, float_format='%.3f')
    # preprocess predicted boxes file
    if boxes_source_path != '':
        samples_ids = annotations_val[[0, 1]].value_counts().keys()
        boxes_val_list = []
        for video_id, time_stamp in samples_ids:
            boxes_val_list.append(boxes_val[np.logical_and(boxes_val[1] == time_stamp, boxes_val[0] == video_id)])
        boxes_val = pd.concat(boxes_val_list)
        for boxes_target_path in boxes_target_paths:
            boxes_val.to_csv(boxes_target_path, header=None, index=False, float_format='%.3f')
    # preprocess frame list file
    frame_list_val = frame_list_val[frame_list_val['original_vido_id'].isin(annotations_val[0].unique())]
    frame_list_val = frame_list_val.replace(np.nan, -1)
    for frame_list_target_path in frame_list_target_paths:
        frame_list_val.to_csv(frame_list_target_path, sep=' ', index=False)


def draw_distributions(source_path, target_path, CLASSES, CLASSES_NAMES, fig_name, NEW_LABEL_ID, original_labels=False):
    dataset = pd.read_csv(source_path, header=None)
    if original_labels:
        class_counts = dataset[dataset[6].isin(CLASSES)][6].value_counts().array
        indices = dataset[dataset[6].isin(CLASSES)][6].value_counts().keys()
        labels = []
        for i in indices:
            labels.append(CLASSES_NAMES[CLASSES.index(i)])
    else:
        class_counts = dataset[dataset[6].isin(NEW_LABEL_ID.values())][6].value_counts().array
        indices = dataset[dataset[6].isin(NEW_LABEL_ID.values())][6].value_counts().keys()
        labels = []
        for i in indices:
            labels.append(CLASSES_NAMES[i-1])
    labels_x = np.arange(len(CLASSES))
    # sns.set_theme(style="whitegrid")
    # sns.barplot(class_counts)
    fig, ax = plt.subplots(figsize=(10, 10), nrows=1, ncols=1)
    fig.patch.set_facecolor('w')
    ax.grid(linestyle='dotted')
    ax.set_facecolor('whitesmoke')
    ax.set_ylabel('number of samples')
    plt.bar(labels_x, class_counts, edgecolor='steelblue', tick_label=labels)
    plt.xticks(rotation=90)
    for x, y in zip(labels_x, class_counts):
        plt.text(x, y + 0.05, '%i' % y, ha='center', va='bottom')
    ax.title.set_text(fig_name)
    plt.show()
    fig.savefig(target_path)
    writer.add_figure(fig_name, fig)


def main(args, CLASSES, NEW_LABEL_ID, CLASSES_NAMES):
    # copy the whole directory
    shutil.copytree(args.base_dir_source, args.base_dir_target, dirs_exist_ok=True)
    if args.dataset == 'ava':
        # draw original sample distribution
        print('#################### Draw Original Distribution ####################')
        source_path = os.path.join(args.base_dir_source, 'annotations/ava_train_v2.2.csv')
        target_path = os.path.join(args.base_dir_source, 'ava_train_v2.2_distribution.png')
        draw_distributions(source_path, target_path, CLASSES, CLASSES_NAMES, 'original sample distribution',
                           NEW_LABEL_ID, original_labels=True)
        source_path = os.path.join(args.base_dir_source, 'annotations/ava_val_v2.2.csv')
        target_path = os.path.join(args.base_dir_source, 'ava_val_v2.2.png')
        draw_distributions(source_path, target_path, CLASSES, CLASSES_NAMES, 'original sample distribution',
                           NEW_LABEL_ID, original_labels=True)

        # preprocess ava_action_list_v2.2.pbtxt
        print('#################### Preprocess ava_action_list_v2.2.pbtxt ####################')
        source_path = os.path.join(args.base_dir_source, 'annotations/ava_action_list_v2.2.pbtxt')
        target_paths = [os.path.join(args.base_dir_target, 'annotations/ava_action_list_v2.2.pbtxt')]
        preprocess_ava_action_list(source_path, target_paths, CLASSES, NEW_LABEL_ID)

        # create ava_train_v2.2.csv & ava_train_predicted_boxes.csv
        print('#################### Preprocess ava_train_v2.2.csv ####################')
        annotations_source_path = os.path.join(args.base_dir_source, 'annotations/ava_train_v2.2.csv')
        annotations_target_path = os.path.join(args.base_dir_target, 'annotations/ava_train_v2.2.csv')
        boxes_source_path = ''
        boxes_target_path = ''
        frame_list_source_path = os.path.join(args.base_dir_source, 'frame_lists/train.csv')
        frame_list_target_path = os.path.join(args.base_dir_target, 'frame_lists/train.csv')
        preprocess_dataset_train(annotations_source_path, annotations_target_path, boxes_source_path, boxes_target_path,
                                 frame_list_source_path, frame_list_target_path, CLASSES.copy(), args.num_samples,
                                 NEW_LABEL_ID, args.exclusive)

        # draw sample distribution after reduction
        print('#################### Draw Training Samples Distribution after Reduction ####################')
        source_path = os.path.join(args.base_dir_target, 'annotations/ava_train_v2.2.csv')
        target_path = os.path.join(args.base_dir_target, 'ava_train_v2.2_distribution.png')
        draw_distributions(source_path, target_path, CLASSES, CLASSES_NAMES, 'sample distribution after reduction', NEW_LABEL_ID)

        # create ava validation dataset
        print('#################### Preprocess ava_val_v2.2.csv ####################')
        annotations_source_path = os.path.join(args.base_dir_source, 'annotations/ava_val_v2.2.csv')
        annotations_target_paths = [os.path.join(args.base_dir_target, 'annotations/ava_val_v2.2.csv')]
        boxes_source_path = ''
        boxes_target_paths = []
        frame_list_source_path = os.path.join(args.base_dir_source, 'frame_lists/val.csv')
        frame_list_target_paths = [os.path.join(args.base_dir_target, 'frame_lists/val.csv')]
        preprocess_dataset_val(annotations_source_path, annotations_target_paths, boxes_source_path, boxes_target_paths,
                               frame_list_source_path, frame_list_target_paths, CLASSES, NEW_LABEL_ID, args.exclusive)

        # draw sample distribution
        print('#################### Draw Validation Samples Distribution after Reduction ####################')
        source_path = os.path.join(args.base_dir_target, 'annotations/ava_val_v2.2.csv')
        target_path = os.path.join(args.base_dir_target, 'ava_val_v2.2_distribution.png')
        draw_distributions(source_path, target_path, CLASSES, CLASSES_NAMES, 'sample distribution after reduction', NEW_LABEL_ID)

    elif args.dataset == 'kinetics':
        # draw original sample distribution
        print('#################### Draw Kinetics Original Distribution ####################')
        source_path = os.path.join(args.base_dir_source, 'annotations/kinetics_train.csv')
        target_path = os.path.join(args.base_dir_source, 'kinetics_train_distribution.png')
        draw_distributions(source_path, target_path, CLASSES, CLASSES_NAMES, 'original sample distribution',
                           NEW_LABEL_ID, original_labels=True)
        source_path = os.path.join(args.base_dir_source, 'annotations/kinetics_val.csv')
        target_path = os.path.join(args.base_dir_source, 'kinetics_val.png')
        draw_distributions(source_path, target_path, CLASSES, CLASSES_NAMES, 'original sample distribution',
                           NEW_LABEL_ID, original_labels=True)

        print('#################### Preprocess kinetics_action_list_v2.2.pbtxt ####################')
        source_path = os.path.join(args.base_dir_source, 'annotations/ava_action_list_v2.2.pbtxt')
        target_paths = [os.path.join(args.base_dir_target, 'annotations/kinetics_action_list_v2.2.pbtxt')]
        preprocess_ava_action_list(source_path, target_paths, CLASSES, NEW_LABEL_ID)

        # create kinetics_train_v2.2.csv & kinetics_train_predicted_boxes.csv with different limit of sample numbers
        print('#################### Preprocess kinetics_train.csv ####################')
        annotations_source_path = os.path.join(args.base_dir_source, 'annotations/kinetics_train.csv')
        annotations_target_path = os.path.join(args.base_dir_target, 'annotations/kinetics_train.csv')
        boxes_source_path = ''
        boxes_target_path = ''
        frame_list_source_path = os.path.join(args.base_dir_source, 'frame_lists/train.csv')
        frame_list_target_path = os.path.join(args.base_dir_target, 'frame_lists/train.csv')
        preprocess_dataset_train(annotations_source_path, annotations_target_path, boxes_source_path, boxes_target_path,
                                 frame_list_source_path, frame_list_target_path, CLASSES.copy(), args.num_samples,
                                 NEW_LABEL_ID, args.exclusive)

        # draw sample distribution after reduction
        print('#################### Draw Training Samples Distribution after Reduction ####################')
        source_path = os.path.join(args.base_dir_target, 'annotations/kinetics_train.csv')
        target_path = os.path.join(args.base_dir_target, 'annotations/kinetics_train_distribution.png')
        draw_distributions(source_path, target_path, CLASSES, CLASSES_NAMES, 'sample distribution after reduction',
                           NEW_LABEL_ID)

        # create kinetics validation dataset
        print('#################### Preprocess kinetics_val.csv ####################')
        annotations_source_path = os.path.join(args.base_dir_source, 'annotations/kinetics_val.csv')
        annotations_target_paths = [os.path.join(args.base_dir_target, 'annotations/kinetics_val.csv')]
        boxes_source_path = ''
        boxes_target_paths = []
        frame_list_source_path = os.path.join(args.base_dir_source, 'frame_lists/val.csv')
        frame_list_target_paths = [os.path.join(args.base_dir_target, 'frame_lists/val.csv')]
        preprocess_dataset_val(annotations_source_path, annotations_target_paths, boxes_source_path, boxes_target_paths,
                               frame_list_source_path, frame_list_target_paths, CLASSES, NEW_LABEL_ID, args.exclusive)

        # draw sample distribution
        print('#################### Draw Validation Samples Distribution after Reduction ####################')
        source_path = os.path.join(args.base_dir_target, 'annotations/kinetics_val.csv')
        target_path = os.path.join(args.base_dir_target, 'kinetics_val_distribution.png')
        draw_distributions(source_path, target_path, CLASSES, CLASSES_NAMES, 'sample distribution after reduction',
                           NEW_LABEL_ID)

    elif args.dataset == 'armasuisse':
        # draw original sample distribution
        print('#################### Draw Original Distribution ####################')
        source_path = os.path.join(args.base_dir_source, 'annotations/armasuisse_train_v2.1.csv')
        target_path = os.path.join(args.base_dir_source, 'armasuisse_train_v2.1_distribution.png')
        draw_distributions(source_path, target_path, CLASSES, CLASSES_NAMES, 'original sample distribution',
                           NEW_LABEL_ID, original_labels=True)
        source_path = os.path.join(args.base_dir_source, 'annotations/armasuisse_val_v2.1.csv')
        target_path = os.path.join(args.base_dir_source, 'armasuisse_val_v2.1.png')
        draw_distributions(source_path, target_path, CLASSES, CLASSES_NAMES, 'original sample distribution',
                           NEW_LABEL_ID, original_labels=True)

        # preprocess armasuisse_action_list_v2.2.pbtxt
        print('#################### Preprocess armasuisse_action_list_v2.2.pbtxt ####################')
        source_path = os.path.join(args.base_dir_source, 'annotations/ava_action_list_v2.2.pbtxt')
        target_paths = [os.path.join(args.base_dir_target, 'annotations/armasuisse_action_list_v2.2.pbtxt')]
        preprocess_ava_action_list(source_path, target_paths, CLASSES, NEW_LABEL_ID)

        # create armasuisse_train_v2.1.csv with different limit of sample numbers
        print('#################### Preprocess ava_train_v2.2.csv ####################')
        annotations_source_path = os.path.join(args.base_dir_source, 'annotations/armasuisse_train_v2.1.csv')
        annotations_target_path = os.path.join(args.base_dir_target, 'annotations/armasuisse_train_v2.1.csv')
        boxes_source_path = ''
        boxes_target_path = ''
        frame_list_source_path = os.path.join(args.base_dir_source, 'frame_lists/train.csv')
        frame_list_target_path = os.path.join(args.base_dir_target, 'frame_lists/train.csv')
        preprocess_dataset_train(annotations_source_path, annotations_target_path, boxes_source_path, boxes_target_path,
                                 frame_list_source_path, frame_list_target_path, CLASSES.copy(), args.num_samples,
                                 NEW_LABEL_ID, args.exclusive)

        # draw sample distribution after reduction
        print('#################### Draw Training Samples Distribution after Reduction ####################')
        source_path = os.path.join(args.base_dir_target, 'annotations/armasuisse_train_v2.1.csv')
        target_path = os.path.join(args.base_dir_target, 'armasuisse_train_v2.1_distribution.png')
        draw_distributions(source_path, target_path, CLASSES, CLASSES_NAMES, 'sample distribution after reduction',
                           NEW_LABEL_ID)

        # create armasuisse validation dataset
        print('#################### Preprocess armasuisse_val_v2.2.csv ####################')
        annotations_source_path = os.path.join(args.base_dir_source, 'annotations/armasuisse_val_v2.1.csv')
        annotations_target_paths = [os.path.join(args.base_dir_target, 'annotations/armasuisse_val_v2.1.csv')]
        boxes_source_path = ''
        boxes_target_paths = []
        frame_list_source_path = os.path.join(args.base_dir_source, 'frame_lists/val.csv')
        frame_list_target_paths = [os.path.join(args.base_dir_target, 'frame_lists/val.csv')]
        preprocess_dataset_val(annotations_source_path, annotations_target_paths, boxes_source_path, boxes_target_paths,
                               frame_list_source_path, frame_list_target_paths, CLASSES, NEW_LABEL_ID, args.exclusive)

        # draw sample distribution
        print('#################### Draw Validation Samples Distribution after Reduction ####################')
        source_path = os.path.join(args.base_dir_target, 'annotations/armasuisse_val_v2.1.csv')
        target_path = os.path.join(args.base_dir_target, 'armasuisse_val_v2.1_distribution.png')
        draw_distributions(source_path, target_path, CLASSES, CLASSES_NAMES, 'sample distribution after reduction',
                           NEW_LABEL_ID)


if __name__ == '__main__':

    # get parameters
    parser = argparse.ArgumentParser(description="Dataset Preprocessing")

    parser.add_argument('--dataset', default='kinetics', help='choose dataset from: ava, kinetics and armasuisse')
    parser.add_argument('--base_dir_source', type=str,
                        default='/srv/beegfs-benderdata/scratch/da_action/data/datasets_yifan/kinetics/',
                        help='directory containing original annotations and frame lists')
    parser.add_argument('--base_dir_target', type=str,
                        default='/srv/beegfs-benderdata/scratch/da_action/data/datasets_yifan/kinetics_5_5000_all/',
                        help='directory to save reduced annotations and frame lists')
    parser.add_argument('--num_samples', type=int, default=5000, help='maximum number of samples for each class')
    parser.add_argument('--exclusive', action='store_true', help='classes are exclusive from each other')
    args = parser.parse_args()

    ###############################
    # MODIFY PARAMETERS BELOW !!! #
    ###############################

    CLASSES = [59, 58, 56, 14, 12]
    NEW_LABEL_ID = {
            59: 2,
            58: 3,
            56: 1,
            14: 7,
            12: 8
        }
    CLASSES_NAMES = ['take a photo', 'touch (an object)', 'throw', 'leave-baggage-unattended',
                     'carrying-baggage', 'drop-baggage', 'walk', 'stand']

    # CLASSES = [7, 13, 16, 2, 3]    # class labels in the original dataset
    # NEW_LABEL_ID = {               # mapping from original class labels to new class labels
    #     7: 1,
    #     13: 2,
    #     16: 3,
    #     2: 4,
    #     3: 5
    # }
    # CLASSES_NAMES = ['touch (an object)', 'throw', 'take a photo', 'walk', 'stand']    # class names from original dataset

    # 3 classes dataset
    # if args.dataset == 'ava' or args.dataset == 'kinetics':
    #     CLASSES = [59, 58, 56]
    #     NEW_LABEL_ID = {59: 1,
    #                     58: 2,
    #                     56: 3}
    # elif args.dataset == 'armasuisse':
    #     CLASSES = [7, 13, 16]
    #     NEW_LABEL_ID = {7: 1,
    #                     13: 2,
    #                     16: 3}
    # CLASSES_NAMES = ['touch (an object)', 'throw', 'take a photo']

    # 5 classes dataset
    # if args.dataset == 'ava' or args.dataset == 'kinetics':
    #     CLASSES = [59, 58, 56, 14, 12]
    #     NEW_LABEL_ID = {
    #         59: 1,
    #         58: 2,
    #         56: 3,
    #         14: 4,
    #         12: 5
    #     }
    # elif args.dataset == 'armasuisse':
    #     CLASSES = [7, 13, 16, 2, 3]
    #     NEW_LABEL_ID = {
    #         7: 1,
    #         13: 2,
    #         16: 3,
    #         2: 4,
    #         3: 5
    #     }
    # CLASSES_NAMES = ['touch (an object)', 'throw', 'take a photo', 'walk', 'stand']

    # 6 classes dataset
    # CLASSES = [1, 8, 10, 11, 12, 14]
    # CLASSES_NAMES = ['bend/bow', 'lie/sleep', 'run/jog', 'sit', 'stand', 'walk']
    # NEW_LABEL_ID = {
    #     1: 1,
    #     8: 2,
    #     10: 3,
    #     11: 4,
    #     12: 5,
    #     14: 6
    # }

    # 9 classes dataset
    # CLASSES = [1, 8, 11, 14, 17, 59, 74, 79, 80]
    # CLASSES_NAMES = ['bend/bow', 'lie/sleep', 'sit', 'walk', 'carry/hold', 'touch', 'listen to',
    #                  'talk to', 'watch']
    # NEW_LABEL_ID = {
    #     1: 1,
    #     8: 2,
    #     11: 3,
    #     14: 4,
    #     17: 5,
    #     59: 6,
    #     74: 7,
    #     79: 8,
    #     80: 9,
    # }

    # 12 classes dataset
    # CLASSES = [1, 17, 18, 7, 36, 47, 10, 12, 56, 58, 59, 14]
    # CLASSES_NAMES = ['bend/bow', 'carry/hold', 'catch', 'jump/leap', 'lift/pick up', 'put down', 'run/jog',
    #                  'stand', 'take a photo', 'throw', 'touch', 'walk']
    # NEW_LABEL_ID = {
    #      1: 1,
    #     17: 2,
    #     18: 3,
    #      7: 4,
    #     36: 5,
    #     47: 6,
    #     10: 7,
    #     12: 8,
    #     56: 9,
    #     58: 10,
    #     59: 11,
    #     14: 12
    # }

    # 15 classes dataset
    # CLASSES = [1, 17, 18, 7, 36, 74, 47, 10, 12, 78, 56, 79, 58, 59, 14]
    # CLASSES_NAMES = ['bend/bow', 'carry/hold', 'catch', 'jump/leap', 'lift/pick up', 'listen to', 'put down', 'run/jog',
    #                  'stand', 'take from', 'take a photo', 'talk to', 'throw', 'touch', 'walk']
    # NEW_LABEL_ID = {
    #      1: 1,
    #     17: 2,
    #     18: 3,
    #      7: 4,
    #     36: 5,
    #     74: 6,
    #     47: 7,
    #     10: 8,
    #     12: 9,
    #     78: 10,
    #     56: 11,
    #     79: 12,
    #     58: 13,
    #     59: 14,
    #     14: 15
    # }

    print('DATASET is ::', args.dataset)
    print('SOURCE DIR is ::', args.base_dir_source)
    print('TARGET DIR is ::', args.base_dir_target)
    print('NUM_SAMPLES is ::', args.num_samples)
    print('CLASSES is ::', CLASSES)
    print('CLASSES_NAMES is ::', CLASSES_NAMES)
    main(args, CLASSES, NEW_LABEL_ID, CLASSES_NAMES)
    print("Finish!")
