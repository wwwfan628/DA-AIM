import os, argparse
from joblib import delayed
from joblib import Parallel
import math
def tar_a_batch(data_dir, tar_dir, batch_id, batch):
    
    cmd = 'tar -cf {:s}/tarball_{:d}.tar '.format(tar_dir, batch_id) 
    ## If using file name with os.walk then add filenames to a text file 
    ## use tar -cv -T file_list.txt -f tarball_{:d}.tar
    for dir_name in batch:
        cmd += './'+dir_name + ' '

    os.system(cmd)

def make_batchs(dirs, nb):
    batchs = []
    batch = []
    batch_size = math.ceil(len(dirs)/nb)
    print('Maximum batch size is goin to be:: ', batch_size)
    for iter, dir in enumerate(dirs):
        batch.append(dir)
        if iter>0 and iter%batch_size == 0:
            batchs.append(batch) 
            batch = []
    
    batchs.append(batch)
    
    return batchs


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='extract frame from videos')
    p.add_argument('data_dir', type=str,
                   help='directory containing multiple directories or files')
    p.add_argument('tar_dir', type=str,
                   help='Where multiple tarballs are going to be saved.')
    p.add_argument('--num_batchs', type=int, default=64,
                   help='Number of directories/files in single tarball')
    p.add_argument('--num_jobs', type=int, default=16,
                   help='Number of parallel jobs to run')
    args = p.parse_args()
    
    print('SOURCE DIR is ::', args.data_dir)
    print('TARGET DIR is ::', args.tar_dir)

    files_or_dirs = os.listdir(args.data_dir)
    
    # files_or_dirs = result = [os.path.join(dp, f) for dp, dn, filenames in os.walk(args.data_dir) for f in filenames if os.path.splitext(f)[1] == '.jpg']
    
    print('Number of directories/files :::>', len(files_or_dirs))

    batchs = make_batchs(files_or_dirs, args.num_batchs)
    
    print('NUmber of batchs', len(batchs))

    if not os.path.isdir(args.tar_dir):
        os.makedirs(args.tar_dir)
    
    owd = os.getcwd()
    
    #first change dir to build_dir path
    os.chdir(args.data_dir)
    ## run parallel packing jobs
    status_lst = Parallel(n_jobs=args.num_jobs)(delayed(tar_a_batch)(args.data_dir, args.tar_dir, batch_id, batch) for batch_id, batch in enumerate(batchs))
    os.chdir(owd)

