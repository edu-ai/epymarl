import os
import glob
import re

if __name__ == "__main__":
    is_communicating = input("Communicating? (true/false): ").lower().strip() == 'true'
    if is_communicating:
        message_shape = input("Message shape: ")
        trajectory_length = input("Trajectory length: ")
    algo = input("Algorithm: ")
    env_name = input("Environment name: ")
    seed = input("seed: ")
    print("#"*50)
    
    sacred_file_path = f"/mnt/d/Zhi Long/results_sacred/epymarl/results/sacred/{algo}/{env_name}/"
    sacred_relevant_fnames = sorted(glob.glob(sacred_file_path+"*/cout.txt"))
    relevant_fnames = []
    for fname in sorted(sacred_relevant_fnames):
        with open(fname, 'r') as f:
            txt = f.read()
            if is_communicating and ("'allow_communications': True" not in txt):
                continue
            elif not is_communicating and ("'allow_communications': True" in txt):
                continue
            if is_communicating and ((message_shape != '' and f"'message_shape': {message_shape}," not in txt) or (trajectory_length != '' and f"'trajectory_length': {trajectory_length}," not in txt)):
                continue
            if seed != '' and not (f"'seed': {seed}," in txt):
                continue

            relevant_fnames.append(fname)

    slurm_file_path = f"/mnt/d/Zhi Long/slurm_out/epymarl/slurm_out/"
    slurm_relevant_fnames = sorted(glob.glob(slurm_file_path+'slurm-*.out'))
    message_types = {}
    trajectory_lengths = {}
    for fname in relevant_fnames:
        with open(fname, 'r') as f:
            lines = f.readlines()
            txt = f.read()
        tb_log_name = None
        tb_log_name_original = None
        slurm_fname = None
        for line in lines:
            match = re.search('Saving models to ', line)
            if match:
                line = line.strip()
                line = line[re.search('results/models/', line).end():].split('/')[0]
                tb_log_name_original = line
                tb_log_name = line.replace(':', '_')
                break
        for sname in slurm_relevant_fnames:
            with open(sname, 'r') as s:
                data = s.read()
                if tb_log_name_original in data:
                    slurm_fname = sname
                    break
        if slurm_fname is not None:
            print(f"SACRED_FNAME: {fname}\t|\tTB_LOG_NAME: {tb_log_name}\t|\tSLURM_FNAME: {slurm_fname}")
            for line in lines:
                message_match = re.search("'message_shape': ", line)
                if message_match:
                    print(line.strip())
                trajectory_match = re.search("'trajectory_length': ", line)
                if trajectory_match:
                    print(line.strip())
            print('')
            








    