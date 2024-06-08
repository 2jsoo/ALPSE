## Import Library
from import_library import *
random.seed(42)
torch.set_num_threads(30)

"""
Data segmentation
    - segment length : 2688points(26.88s;100Hz)
    - sliding window : 2688points(26.88s)
"""

folder_name = 'final_data'

@ray.remote
def preprocessing_labeling(key, label):
    """
    Label annotation 
        reference paper : Automatic segmentation of atrial fibrillation and flutter in single-lead electrocardiograms by self-supervised learning and Transformer architecture (JAMIA, 2023)
    1. Making a suspected AF/AFL pseudo-label for a single data point when AF/AFL was present in greater than 50% of surrounding 10 second intervals
    2. If the instance that the pseudo-label persisted for a duration exceeding 10 seconds, it was then labeled as AF/AFL. 
    """
    smooth_label = dict()
    suspected_pseudolbl = []
    fs = 100
    for idx, lbl in enumerate(label):
        start = max(0, idx - 5*fs)
        end = min(len(label), idx + 5*fs)
        window_arr = label[start:end]

        suspected_pseudolbl.append(0 if np.sum(window_arr == 0) > fs * 5 else 1)

    final_lbl = []
    consider_before_duration = 0
    consider_before_lbl = ''
    cur_lbl_idx = 0
    while(True):        
        cur_lbl = suspected_pseudolbl[cur_lbl_idx]

        try:
            new_lbl_idx = suspected_pseudolbl[cur_lbl_idx:].index(cur_lbl^1) + cur_lbl_idx
        except ValueError:
            new_lbl_idx = len(suspected_pseudolbl)
        
        if consider_before_lbl == cur_lbl:
            cur_lbl_duration = new_lbl_idx - cur_lbl_idx + consider_before_duration 
        else:
            cur_lbl_duration = new_lbl_idx - cur_lbl_idx

        if cur_lbl_duration > 10 * fs:
            consider_before_duration = 0
            consider_before_lbl = cur_lbl
            final_lbl.extend([cur_lbl]*(new_lbl_idx - cur_lbl_idx))
            cur_lbl_idx = new_lbl_idx
        else:
            consider_before_duration = cur_lbl_duration
            if len(final_lbl) == 0:
                consider_before_lbl = cur_lbl^1
            else:
                consider_before_lbl = final_lbl[-1]
            final_lbl.extend([consider_before_lbl]*(new_lbl_idx - cur_lbl_idx))
            cur_lbl_idx = new_lbl_idx

        if cur_lbl_idx >= len(suspected_pseudolbl) - 1:
            break
    
    smooth_label[key] = final_lbl

    return smooth_label

@ray.remote
def preprocessing_data_xmidwidth(key, data, label, seg_length):
    """
    Label construction
    label : [class, xmid, width]
        - class : AF/AFL(0), Others(1)
        - xmid : central point of class interval
        - width : duration of class interval
    """


    final_data = dict()
    final_label = dict()    

    for lead_name in data.keys():    

        data[lead_name] = np.asarray(data[lead_name])
        label = np.asarray(label)
        
        # Data chunking : 26.88sec (2688 points)
        n = int(seg_length * 100)
        sliding_window = int(seg_length * 100)
        data_segments = [data[lead_name][x:x+n] for x in range(0, len(data[lead_name])-n, sliding_window)]
        label_segments = [label[x:x+n] for x in range(0, len(label)-n, sliding_window)]
        

        seg_num = 1       
        for data_seg, label_seg in tqdm(zip(data_segments, label_segments)):
            label_seg = list(label_seg)

            new_label_segments = []
            cur_lbl_idx = 0
            while(True):
                cur_lbl = label_seg[cur_lbl_idx]
                
                try:
                    new_lbl_idx = label_seg[cur_lbl_idx:].index(cur_lbl^1) 
                    new_lbl_idx += cur_lbl_idx
                except ValueError:
                    new_lbl_idx = len(label_seg)-1

                xmid = (new_lbl_idx + cur_lbl_idx) / 2
                width = new_lbl_idx - cur_lbl_idx
                new_label_segments.append([cur_lbl, xmid, width])

                cur_lbl_idx = new_lbl_idx
                
                if cur_lbl_idx >= len(label_seg) - 1:
                    break

            ### Data normalization
            data_seg = (data_seg - min(data_seg)) / (max(data_seg) - min(data_seg))


            #### Remove invalid data
            try:
                _, rpeaks = nk.ecg_peaks(data_seg, sampling_rate=100)
            except:
                continue
            rpeaks = rpeaks['ECG_R_Peaks']
            if len(rpeaks) <= 10:
                continue
                    
            final_data[key+f'_{lead_name}_{seg_num}'] = data_seg
            final_label[key+f'_{lead_name}_{seg_num}'] = new_label_segments
            seg_num += 1
            
    return final_data, final_label


list_database = ['ltafdb', 'afdb', 'cpsc2021', 'nsrdb', 'mitdb']
list_seg_length = [26.88]

for seg_length in list_seg_length:
    for database in list_database:
        with open(f'../{folder_name}/{database}/data.pkl', 'rb') as f:
            data = pickle.load(f)
        with open(f'../{folder_name}/{database}/label.pkl', 'rb') as f:
            label = pickle.load(f)

        ray.init(num_cpus = 8)
        results = ray.get([preprocessing_labeling.remote(key, key_label) for db in label.keys() for key, key_label in tqdm(label[db].items())])
        ray.shutdown()
        new_label = dict()
        for result in results:
            new_label.update(result)
        label = {database: new_label}


        ray.init(num_cpus = 8)
        results = ray.get([preprocessing_data_xmidwidth.remote(key, data[db][key], key_label, seg_length) for db in label.keys() for key, key_label in tqdm(label[db].items())])
        ray.shutdown()

        data = dict()
        label = dict()
        for result in results:
            data.update(result[0])
            label.update(result[1])
        total_data = {database: data}
        total_label = {database: label}

        total_seg = 0
        for db in total_label.keys():
            total_seg += len(total_label[db].keys())
        del data
        del label
        gc.collect()


        os.makedirs(f'{folder_name}/{database}', exist_ok=True)
        with open(f'{folder_name}/{database}/data.pkl', 'wb') as f: 
            pickle.dump(total_data, f)
        with open(f'{folder_name}/{database}/label.pkl', 'wb') as f:
            pickle.dump(total_label, f)