
import os, argparse

def read_classes(base_dir):
    f = open(os.path.join(base_dir, 'moments_categories.txt'),'r')
    classes = {}
    class_names = []
    for line in f:
        if len(line)<3:
            continue
        line = line.rstrip('\n')
        line = line.split(',')
        classes[line[0]] = int(line[1])
        class_names.append(line[0])
    
    return classes, class_names

def main(base_dir, subset='train'):
    classes, class_names = read_classes(base_dir)
    fid = open(os.path.join(base_dir, '{}.csv'.format(subset)),'w')
    for cls in class_names:
        videos_dir = os.path.join(base_dir, subset, cls)
        videos = os.listdir(videos_dir)
        label=classes[cls]
        for vid in videos:
            if vid.endswith('.mp4'):
                fid.write('{:s}/{:s}/{:s} {:d}\n'.format(subset,cls,vid, label))
    

if __name__ == '__main__':
    description = 'Helper script for downloading and trimming kinetics videos.'
    p = argparse.ArgumentParser(description=description)
    p.add_argument('--base_dir', default='/raid/susaha/datasets/mit', type=str,
                   help='Output directory where videos will be saved.')
    p.add_argument('--subset', default='train', type=str,
                   help='Output directory where videos will be saved.')

    args = p.parse_args()
    main(args.base_dir, args.subset)