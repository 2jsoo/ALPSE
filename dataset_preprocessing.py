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


def dataset_preprocessing_leadi_ii_iii(path_base, database):
    total_array_aux_note = dict()
    total_array_data = dict()
    print(database)
    
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
            lowcut = 0.5
            highcut = 40
            filtered_valid_data = butter_bandpass_filter(valid_data, lowcut, highcut, fs, order=5)

            ## resample 
            target_fs = 100
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


path_base = 'datasets'

for database in ['ltafdb', 'afdb', 'cpsc2021', 'nsrdb', 'mitdb']:
    total_array_data, total_array_aux_note = dataset_preprocessing_leadi_ii_iii(path_base, database)    

    print(database, len(total_array_aux_note[database].keys()), 'files')

    os.makedirs(f'final_data/{database}', exist_ok=True)
    with open(f'final_data/{database}/data.pkl', 'wb') as f:
        pickle.dump(total_array_data, f)
    with open(f'final_data/{database}/label.pkl', 'wb') as f:
        pickle.dump(total_array_aux_note, f)