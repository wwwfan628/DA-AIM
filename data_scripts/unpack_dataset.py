import os, argparse
from joblib import delayed
from joblib import Parallel

def untar(tar_dir, tarball, data_dir):
    print(tar_dir,tarball, data_dir)
    cmd = 'tar -xf {:s} -C {:s}'.format(os.path.join(tar_dir, tarball), data_dir) 
    os.system(cmd)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='extract frame from videos')
    p.add_argument('data_dir', type=str,
                   help='directory where to untar the tar balls')
    p.add_argument('tar_dir', type=str,
                   help='Where multiple tarballs are present.')
    p.add_argument('--num_jobs', type=int, default=8,
                   help='Number of parallel jobs to run')
    args = p.parse_args()
    
    print('Source directory is ', args.tar_dir)
    print('Target directory is ', args.data_dir)
    
    tar_files = os.listdir(args.tar_dir)
    tar_files = [f for f in tar_files if f.endswith('.tar')]

    print('Number of tar balls are :::>', len(tar_files))

    if not os.path.isdir(args.data_dir):
        os.makedirs(args.data_dir)

    status_lst = Parallel(n_jobs=args.num_jobs)(delayed(untar)( args.tar_dir, tarball, args.data_dir) for tarball in tar_files)


