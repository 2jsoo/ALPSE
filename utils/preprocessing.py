from import_library import *

def butter_bandpass(lowcut, highcut, fs, order=5): ## code reference : https://alice-secreta.tistory.com/23
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5): ## code reference : https://alice-secreta.tistory.com/23
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = signal.lfilter(b, a, data)
    return y

def dataset_preprocessing_leadi_ii_iii(path_base, database, target_fs, lowcut, highcut):
    total_array_aux_note = dict()
    total_array_data = dict()
    
    if database != 'cpsc2021':
        path_database = path_base + f'/{database.split("_")[0]}/1.0.0'
        subjects = [a.split(".")[0] for a in os.listdir(path_database) if a.endswith('.hea')]
    else:
        path_database = path_base + f'/{database.split("_")[0]}/1.0.0'
        subjects = []
        for sub_path in ['Training_set_I', 'Training_set_II']:
            if len(subjects) == 0:
                subjects = [f'{sub_path}/{a.split(".")[0]}' for a in os.listdir(path_database + '/' + sub_path) if a.endswith('.hea')]
            else:
                subjects.extend([f'{sub_path}/{a.split(".")[0]}' for a in os.listdir(path_database + '/' + sub_path) if a.endswith('.hea')])

    array_aux_note = dict()
    array_data = dict()
    for subject in tqdm(subjects):
        try:
            ann = wfdb.rdann(os.path.join(path_database, subject), 'atr').__dict__
        except FileNotFoundError:
            print('FileNotFoundError', subject)
            continue
        try:
            data = wfdb.rdsamp(os.path.join(path_database, subject))
        except ValueError:
            print('ValueError', subject)
            continue
        
        fs = data[1]['fs'] ## sampling frequency
        lead_list = data[1]['sig_name'] ## ECG lead name

        #### Select lead I, II, III and if not avaiable, select first lead
        if len(list(set(lead_list)&set(['I', 'II', 'MLII', 'III']))) == 0:
            idx_mlii = [0]
        else:
            idx_mlii = []
            for lead_ in list(set(lead_list)&set(['I', 'II', 'MLII', 'III'])):
                idx_mlii.append(lead_list.index(lead_))
            
        
        two_lead_data = dict()
        for lead_num in idx_mlii:
            valid_data = data[0][:, lead_num]
            lead_name = data[1]['sig_name'][lead_num]
            t_aux_note = np.asarray(ann['aux_note'])
            t_sample = np.asarray(ann['sample'])

            #### Change unavailable aux_note(ex. MISSB, PSE, \x01 Aux, M, P, T, TS, None) to ''
            t_aux_note = np.where((t_aux_note == 'MISSB') | 
                                  (t_aux_note == 'PSE') | 
                                  (t_aux_note == "\x01 Aux") |
                                  (t_aux_note == 'M') | 
                                  (t_aux_note == 'P') | 
                                  (t_aux_note == 'T') | 
                                  (t_aux_note == 'TS') | 
                                  (t_aux_note == 'None'), '', t_aux_note)
            
            #### Remove unavailable data
            if len(set(t_aux_note)) == 1 and list(set(t_aux_note))[0] == '':
                pass
            else:
                valid_aux_idx = np.where(t_aux_note != '')[0]
                first_valid_aux_idx = valid_aux_idx[0]
                first_valid_aux_samplepoint = t_sample[first_valid_aux_idx]
                valid_data = valid_data[first_valid_aux_samplepoint:]
                t_aux_note = t_aux_note[first_valid_aux_idx:]
                t_sample = t_sample[first_valid_aux_idx:]
                t_sample = t_sample - first_valid_aux_samplepoint


            """ 
            Filtering reference : 
                Paper : A generalizable and robust deep learning method for atrial fibrillation detection from long-term electrocardiogram. Biomedical Signal Processing and Control (Biomedical Signal Processing and Control, 2024, Zou, et al.) 
                - Remove baseline drift and high-frequency noise
                    - A fifth-order Butterworth filter with a bandpass frequency range of 0.5 ∼ 40 Hz
            """
            # lowcut = 0.5
            # highcut = 40
            filtered_valid_data = butter_bandpass_filter(valid_data, lowcut, highcut, fs, order=5)

            ## resample 
            # target_fs = 100
            resampled_valid_data, resampled_location = processing.resample_sig(filtered_valid_data, fs=fs, fs_target=target_fs)
            resampled_t_sample = processing.resample_ann(t_sample, fs=fs, fs_target=target_fs)

            ###### aux note interpolation => Assign aux_note to each sample point
            ## AFIB or AFL : 0 //// Others : 1
            new_aux_note = []
            tmp = ''
            start_point = 0
            for idx, aux_note in enumerate(t_aux_note): 
                aux_note = aux_note.split("(")[-1] # aux_note : ex. '(AFIB', '(AFL' 
                if idx == 0:
                    cur_point = resampled_t_sample[idx]
                    if len(t_aux_note) == 1: # Single aux_note
                        if cur_point != 0: 
                            new_aux_note.extend([1]*(cur_point)) 

                            start_point = cur_point

                            if aux_note in ['AFIB', 'AFL']: 
                                new_aux_note.extend([0] * len(resampled_valid_data))
                            else:
                                new_aux_note.extend([1] * len(resampled_valid_data))
                        else: 
                            if aux_note in ['AFIB', 'AFL']:
                                new_aux_note.extend([0] * len(resampled_valid_data))
                            else:
                                new_aux_note.extend([1] * len(resampled_valid_data))  
                        break 
                    else: # Multiple aux_note
                        if cur_point != 0: 
                            new_aux_note.extend([1]*(cur_point))
                            
                        tmp = aux_note
                        start_point = cur_point
                elif idx != len(t_aux_note)-1: 
                    cur_point = resampled_t_sample[idx]
                    if (aux_note == tmp) or (aux_note == ''): # same as before
                        pass
                    else: # not same as before
                        if tmp in ['AFIB', 'AFL']:
                            new_aux_note.extend([0] * (cur_point - start_point))
                        else:
                            new_aux_note.extend([1] * (cur_point - start_point))
                        
                        start_point = cur_point
                        tmp = aux_note
                else: # last aux_note
                    cur_point = resampled_t_sample[idx]
                    if (aux_note == tmp) or (aux_note == ''):
                        if tmp in ['AFIB', 'AFL']:
                            new_aux_note.extend([0] * (len(resampled_valid_data) - start_point))
                        else:
                            new_aux_note.extend([1] * (len(resampled_valid_data) - start_point))

                    else:
                        if cur_point == len(resampled_valid_data)-1:
                            if tmp in ['AFIB', 'AFL']:
                                new_aux_note.extend([0] * (cur_point - start_point))
                            else:
                                new_aux_note.extend([1] * (cur_point - start_point))

                            if aux_note in ['AFIB', 'AFL']:
                                new_aux_note.extend([0])
                            else:
                                new_aux_note.extend([1])

                        else:
                            if tmp in ['AFIB', 'AFL']:
                                new_aux_note.extend([0] * (cur_point - start_point))
                            else:
                                new_aux_note.extend([1] * (cur_point - start_point))

                            start_point = cur_point
                            if aux_note in ['AFIB', 'AFL']:
                                new_aux_note.extend([0] * (len(resampled_valid_data) - start_point))
                            else:
                                new_aux_note.extend([1] * (len(resampled_valid_data) - start_point))

            if len(new_aux_note) != len(resampled_valid_data):
                new_aux_note = new_aux_note[:len(resampled_valid_data)]

            new_aux_note = np.asarray(new_aux_note)

            two_lead_data[lead_name] = resampled_valid_data
            

        array_aux_note.update({subject : new_aux_note}) 
        array_data.update({subject : two_lead_data})

    total_array_aux_note[database] = array_aux_note
    total_array_data[database] = array_data
        
    return total_array_data, total_array_aux_note

@ray.remote
def preprocessing_labeling(key, label, fs):
    """
    Label annotation 
        reference paper : Automatic segmentation of atrial fibrillation and flutter in single-lead electrocardiograms by self-supervised learning and Transformer architecture (JAMIA, 2023)
    1. Making a suspected AF/AFL pseudo-label for a single data point when AF/AFL was present in greater than 50% of surrounding 10 second intervals
    2. If the instance that the pseudo-label persisted for a duration exceeding 10 seconds, it was then labeled as AF/AFL. 
    """
    smooth_label = dict()
    suspected_pseudolbl = []
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
def preprocessing_data_xmidwidth(key, data, label, seg_length, fs):
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
        n = int(seg_length * fs)
        sliding_window = int(seg_length * fs)
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

def make_kfold_data(path, train_db, seg_length, seed):
    data = pd.read_pickle(f'{path}/{train_db}/data_{seg_length}s.pkl')
    label = pd.read_pickle(f'{path}/{train_db}/label_{seg_length}s.pkl')

    if train_db != 'cpsc2021': # subject_leadname_segnum
        subjects = np.array(list(set([a.split("_")[0] for a in list(label[train_db].keys())])))
    else: # Training_set_I(or II)/subject(ex.data_54_1)_leadname_segnum
        subjects = np.array(list(set([f'{a.split("/")[-1].split("_")[0]}_{a.split("/")[-1].split("_")[1]}_{a.split("/")[-1].split("_")[2]}' for a in list(label[train_db].keys())])))

    train_key, test_key = train_test_split(subjects, test_size=0.2, random_state=seed)

    if train_db != 'cpsc2021':
        train_subjects = np.array(list(set([a.split("_")[0] for a in train_key])))
        train_key = [a for a in list(label[train_db].keys()) if a.split("_")[0] in train_key]
        test_key = [a for a in list(label[train_db].keys()) if a.split("_")[0] in test_key]
    else:
        train_subjects = np.array(list(set([f'{a.split("/")[-1].split("_")[0]}_{a.split("/")[-1].split("_")[1]}_{a.split("/")[-1].split("_")[2]}' for a in train_key])))
        train_key = [a for a in list(label[train_db].keys()) if f'{a.split("/")[-1].split("_")[0]}_{a.split("/")[-1].split("_")[1]}_{a.split("/")[-1].split("_")[2]}' in train_key]
        test_key = [a for a in list(label[train_db].keys()) if f'{a.split("/")[-1].split("_")[0]}_{a.split("/")[-1].split("_")[1]}_{a.split("/")[-1].split("_")[2]}' in test_key]

    # Kfold
    SPLITS = 5
    kf = KFold(n_splits = SPLITS, random_state=seed, shuffle=True)
    n_iter = 0
    for train_idx, valid_idx in kf.split(train_subjects):
        train_data = dict()
        train_label = dict()
        valid_data = dict()
        valid_label = dict()
        
        n_iter += 1
        logging.info(f'--------------------{n_iter}Fold-------------------')
        train_key = train_subjects[train_idx]
        valid_key = train_subjects[valid_idx]
        if train_db != 'cpsc2021':
            train_key = [a for a in list(label[train_db].keys()) if a.split("_")[0] in train_key]
            valid_key = [a for a in list(label[train_db].keys()) if a.split("_")[0] in valid_key]
        else:
            train_key = [a for a in list(label[train_db].keys()) if f'{a.split("/")[-1].split("_")[0]}_{a.split("/")[-1].split("_")[1]}_{a.split("/")[-1].split("_")[2]}' in train_key]
            valid_key = [a for a in list(label[train_db].keys()) if f'{a.split("/")[-1].split("_")[0]}_{a.split("/")[-1].split("_")[1]}_{a.split("/")[-1].split("_")[2]}' in valid_key]



        train_summary = {0: 0, 1: 0}
        valid_summary = {0: 0, 1: 0}
        for k in train_key:
            for sub_label in label[train_db][k]:
                if sub_label[0] == 0: # af
                    train_summary[0] += 1
                elif sub_label[0] == 1: # others
                    train_summary[1] += 1
        for k in valid_key:
            for sub_label in label[train_db][k]:
                if sub_label[0] == 0:
                    valid_summary[0] += 1
                elif sub_label[0] == 1:
                    valid_summary[1] += 1

        logging.info(f'Development : {train_summary} Valid : {valid_summary}')

        for k in train_key:
            train_data[f'{train_db}_{k}'] = data[train_db][k]
            train_label[f'{train_db}_{k}'] = label[train_db][k]
        for k in valid_key:
            valid_data[f'{train_db}_{k}'] = data[train_db][k]
            valid_label[f'{train_db}_{k}'] = label[train_db][k]

        os.makedirs(f'{path}/{train_db}/Train/{n_iter}fold', exist_ok=True)
        with open(f'{path}/{train_db}/Train/{n_iter}fold/subject_key_{seg_length}s.pkl', 'wb') as f:
            if train_db != 'cpsc2021':
                pickle.dump(list(set([f'{a.split("_")[0]}_{a.split("_")[1]}' for a in train_label.keys()])), f)
            else:
                pickle.dump(list(set([f'{a.split("/")[-1].split("_")[0]}_{a.split("/")[-1].split("_")[1]}_{a.split("/")[-1].split("_")[2]}' for a in train_label.keys()])), f)
        with open(f'{path}/{train_db}/Train/{n_iter}fold/data{seg_length}s.pkl', 'wb') as f:
            pickle.dump(train_data, f)
        with open(f'{path}/{train_db}/Train/{n_iter}fold/label{seg_length}s.pkl', 'wb') as f:
            pickle.dump(train_label, f)

        os.makedirs(f'{path}/{train_db}/Valid/{n_iter}fold', exist_ok=True)
        with open(f'{path}/{train_db}/Valid/{n_iter}fold/subject_key_{seg_length}s.pkl', 'wb') as f:
            if train_db != 'cpsc2021':
                pickle.dump(list(set([f'{a.split("_")[0]}_{a.split("_")[1]}' for a in valid_label.keys()])), f)
            else:
                pickle.dump(list(set([f'{a.split("/")[-1].split("_")[0]}_{a.split("/")[-1].split("_")[1]}_{a.split("/")[-1].split("_")[2]}' for a in valid_label.keys()])), f)
        with open(f'{path}/{train_db}/Valid/{n_iter}fold/data{seg_length}s.pkl', 'wb') as f:
            pickle.dump(valid_data, f)
        with open(f'{path}/{train_db}/Valid/{n_iter}fold/label{seg_length}s.pkl', 'wb') as f:
            pickle.dump(valid_label, f)


    test_data = dict()
    test_label = dict()
    for k in test_key:
        test_data[f'{train_db}_{k}'] = data[train_db][k]
        test_label[f'{train_db}_{k}'] = label[train_db][k]
    
    test_summary = {0: 0, 1: 0}
    for k in test_key:
        for sub_label in label[train_db][k]:
            if sub_label[0] == 0: # af
                test_summary[0] += 1
            elif sub_label[0] == 1: # others
                test_summary[1] += 1

    logging.info(f'Test : {test_summary}')
    os.makedirs(f'{path}/{train_db}/Test', exist_ok=True)
    with open(f'{path}/{train_db}/Test/subject_key_{seg_length}s.pkl', 'wb') as f:
        if train_db != 'cpsc2021':
            pickle.dump(list(set([f'{a.split("_")[0]}_{a.split("_")[1]}' for a in test_key])), f)
        else:
            pickle.dump(list(set([f'{a.split("/")[-1].split("_")[0]}_{a.split("/")[-1].split("_")[1]}_{a.split("/")[-1].split("_")[2]}' for a in test_key])), f)
    with open(f'{path}/{train_db}/Test/data{seg_length}s.pkl', 'wb') as f:
        pickle.dump(test_data, f)
    with open(f'{path}/{train_db}/Test/label{seg_length}s.pkl', 'wb') as f:
        pickle.dump(test_label, f)

def make_train_test_data(path, train_db, seg_length, seed):
    total_data = pd.read_pickle(f'{path}/{train_db.split("_")[0]}/Train/1fold/data{seg_length}s.pkl')
    total_label = pd.read_pickle(f'{path}/{train_db.split("_")[0]}/Train/1fold/label{seg_length}s.pkl')
    valid_data = pd.read_pickle(f'{path}/{train_db.split("_")[0]}/Valid/1fold/data{seg_length}s.pkl')
    valid_label = pd.read_pickle(f'{path}/{train_db.split("_")[0]}/Valid/1fold/label{seg_length}s.pkl')
    test_data = pd.read_pickle(f'{path}/{train_db.split("_")[0]}/Test/data{seg_length}s.pkl')
    test_label = pd.read_pickle(f'{path}/{train_db.split("_")[0]}/Test/label{seg_length}s.pkl')

    total_data.update(valid_data)
    total_label.update(valid_label)
    total_data.update(test_data)
    total_label.update(test_label)

    if len(train_db.split("_")) > 1:
        with open(f'{path}/{train_db.split("_")[1]}/Train/1fold/data{seg_length}s.pkl', 'rb') as f:
            add_train_data = pickle.load(f)
        with open(f'{path}/{train_db.split("_")[1]}/Train/1fold/label{seg_length}s.pkl', 'rb') as f:
            add_train_label = pickle.load(f)
        with open(f'{path}/{train_db.split("_")[1]}/Valid/1fold/data{seg_length}s.pkl', 'rb') as f:
            add_valid_data = pickle.load(f)
        with open(f'{path}/{train_db.split("_")[1]}/Valid/1fold/label{seg_length}s.pkl', 'rb') as f:
            add_valid_label = pickle.load(f)
        with open(f'{path}/{train_db.split("_")[1]}/Test/data{seg_length}s.pkl', 'rb') as f:
            add_test_data = pickle.load(f)
        with open(f'{path}/{train_db.split("_")[1]}/Test/label{seg_length}s.pkl', 'rb') as f:
            add_test_label = pickle.load(f) 
    
        total_data.update(add_train_data)
        total_label.update(add_train_label)
        total_data.update(add_valid_data)
        total_label.update(add_valid_label)
        total_data.update(add_test_data)
        total_label.update(add_test_label)

        del add_train_data
        del add_train_label
        del add_valid_data
        del add_valid_label
        del add_test_data
        del add_test_label

    del valid_data
    del valid_label
    del test_data
    del test_label
    gc.collect()

    if train_db != 'cpsc2021':
        subjects = np.asarray(list(set([a.split("_")[1] for a in list(total_label.keys())])))
        print(len(subjects), subjects[0])
        train_subjects, test_subjects = train_test_split(subjects, test_size=0.2, random_state=seed)
        train_key = [a for a in list(total_label.keys()) if a.split("_")[1] in train_subjects]
        valid_key = [a for a in list(total_label.keys()) if a.split("_")[1] in test_subjects]
    else:
        subjects = np.asarray(list(set([f'{a.split("/")[-1].split("_")[0]}_{a.split("/")[-1].split("_")[1]}_{a.split("/")[-1].split("_")[2]}' for a in list(total_label.keys())])))
        train_subjects, test_subjects = train_test_split(subjects, test_size=0.2, random_state=seed)
        train_key = [a for a in list(total_label.keys()) if f'{a.split("/")[-1].split("_")[0]}_{a.split("/")[-1].split("_")[1]}_{a.split("/")[-1].split("_")[2]}' in train_subjects]
        valid_key = [a for a in list(total_label.keys()) if f'{a.split("/")[-1].split("_")[0]}_{a.split("/")[-1].split("_")[1]}_{a.split("/")[-1].split("_")[2]}' in test_subjects]

    train_data = dict()
    train_label = dict()
    valid_data = dict()
    valid_label = dict()
    for k in tqdm(train_key):
        train_data[k] = total_data[k]
        train_label[k] = total_label[k]
    for k in tqdm(valid_key):
        valid_data[k] = total_data[k]
        valid_label[k] = total_label[k]
    
    os.makedirs(f'{path}/{train_db}/Total_train_valid', exist_ok=True)
    with open(f'{path}/{train_db}/Total_train_valid/train_data_{seg_length}s.pkl', 'wb') as f:
        pickle.dump(train_data, f)
    with open(f'{path}/{train_db}/Total_train_valid/train_label_{seg_length}s.pkl', 'wb') as f:
        pickle.dump(train_label, f)
    with open(f'{path}/{train_db}/Total_train_valid/valid_data_{seg_length}s.pkl', 'wb') as f:
        pickle.dump(valid_data, f)
    with open(f'{path}/{train_db}/Total_train_valid/valid_label_{seg_length}s.pkl', 'wb') as f:
        pickle.dump(valid_label, f)
