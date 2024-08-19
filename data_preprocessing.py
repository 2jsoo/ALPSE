from import_library import *
from utils.preprocessing import *

parser = argparse.ArgumentParser()
parser.add_argument('config_file',
					type=str,
                    help='Config path')
                    
args = parser.parse_args()

config = configparser.ConfigParser()
config.read(args.config_file)

path_base = config['Data']['path']
save_path = config['Data']['save_path']
database = config['Data']['db']
fs = int(config['Data']['fs'])
seg_length = float(config['Data']['seg_length'])

logging.basicConfig(filename="data_preprocessing.log", filemode="a", level=logging.DEBUG)
logging.info('Data Preprocessing')
logging.info(f'Dataset : {database}')

with open(f'{path_base}/{database}/data.pkl', 'rb') as f:
    data = pickle.load(f)
with open(f'{path_base}/{database}/label.pkl', 'rb') as f:
    label = pickle.load(f)

ray.init(num_cpus = 8)
results = ray.get([preprocessing_labeling.remote(key, key_label, fs) for db in label.keys() for key, key_label in tqdm(label[db].items())])
ray.shutdown()
new_label = dict()
for result in results:
    new_label.update(result)
label = {database: new_label}


ray.init(num_cpus = 8)
results = ray.get([preprocessing_data_xmidwidth.remote(key, data[db][key], key_label, seg_length, fs) for db in label.keys() for key, key_label in tqdm(label[db].items())])
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

os.makedirs(f'{path_base}/{database}', exist_ok=True)
with open(f'{path_base}/{database}/data_{seg_length}s.pkl', 'wb') as f: 
    pickle.dump(total_data, f)
with open(f'{path_base}/{database}/label_{seg_length}s.pkl', 'wb') as f:
    pickle.dump(total_label, f)