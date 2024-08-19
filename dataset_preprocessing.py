from import_library import *
from utils.preprocessing import *

parser = argparse.ArgumentParser()
parser.add_argument('config_file',
					type=str,
                    help='Config path')
                    
args = parser.parse_args()

config = configparser.ConfigParser()
config.read(args.config_file)

path_base = config['Dataset']['path']
save_path = config['Dataset']['save_path']
database = config['Dataset']['db']
target_fs = int(config['Dataset']['target_fs'])
lowcut = float(config['Dataset']['lowcut'])
highcut = float(config['Dataset']['highcut'])


logging.basicConfig(filename="dataset_preprocessing.log", filemode="a", level=logging.DEBUG)
logging.info('Dataset Preprocessing')
logging.info(f'Dataset : {database}')

total_array_data, total_array_aux_note = dataset_preprocessing_leadi_ii_iii(path_base, database, target_fs, lowcut, highcut)    

logging.info(f'{len(total_array_aux_note[database].keys())} Files')

os.makedirs(f'{save_path}/{database}', exist_ok=True)
with open(f'{save_path}/{database}/data.pkl', 'wb') as f:
    pickle.dump(total_array_data, f)
with open(f'{save_path}/{database}/label.pkl', 'wb') as f:
    pickle.dump(total_array_aux_note, f)