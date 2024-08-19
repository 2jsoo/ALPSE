from import_library import *
from dataloader import *
from loss import *
from model import *
from utils.utils import *
from utils.preprocessing import make_train_test_data


parser = argparse.ArgumentParser()
parser.add_argument('config_file',
					type=str,
                    help='Config path')
                    
args = parser.parse_args()

config = configparser.ConfigParser()
config.read(args.config_file)

#################### Load args ####################
path = config['Base']['path']
save_path = config['Base']['save_path'] 
gpu = config['Base']['gpu']
seed = int(config['Base']['seed'])


train_db = config['Data']['train_db'] # afdb
test_db = config['Data']['test_db'] # cpsc2021
fs = int(config['Data']['fs'])
seg_length = float(config['Data']['seg_length'])

S = int(config['Model']['S'])
B = int(config['Model']['B'])
C = int(config['Model']['C'])
out_channels = int(config['Model']['out_channels']) 
activation = config['Model']['activation']

lambda_class = float(config['Loss']['lambda_class'])
lambda_noobj = float(config['Loss']['lambda_noobj'])
lambda_obj = float(config['Loss']['lambda_obj'])
lambda_iou = float(config['Loss']['lambda_iou'])
iou_type = config['Loss']['iou_type']
reduction = config['Loss']['reduction']


BATCH = int(config['SYS']['batch'])
num_epochs = int(config['SYS']['num_epochs'])
optim_name = config['SYS']['optim_name']
early_stop = int(config['SYS']['early_stop'])
MOMENTUM = float(config['SYS']['momentum'])
iou_threshold = float(config['SYS']['iou_threshold'])
nms_iou_threshold = float(config['SYS']['nms_iou_threshold']) 
confidence_threshold = float(config['SYS']['confidence_threshold'])
warmup_lr_initial = float(config['SYS']['lr_init'])
warmup_epochs = int(config['SYS']['warmup_epochs'])
LR = float(config['SYS']['lr'])
WD = float(config['SYS']['WD'])

if iou_type == 'diou':
    diou = True 
else:
    diou = False

############################################################

os.environ["CUDA_VISIBLE_DEVICES"]=gpu 
torch.set_num_threads(30)
np.random.seed(seed) 
DEVICE = "cuda:0" if torch.cuda.is_available else "cpu" 


PATH = f'{save_path}/{train_db}/Internal'

logging.basicConfig(filename="External_validation.log", filemode="w", level=logging.DEBUG)
logging.info(f'Train data : {train_db}')

################################################ Train Test split ################################################
FOLDER_PATH = f'{PATH}/final_model'

if os.path.exists(FOLDER_PATH):
    pass
else:
    os.makedirs(FOLDER_PATH, exist_ok=True)

    logging.info(f'Load Data')
    try:
        pd.read_pickle(f'{path}/{train_db}/Total_train_valid/train_label_{seg_length}s.pkl')
    except FileNotFoundError:
        make_train_test_data(path, train_db, seg_length, seed)
    
    train_data = pd.read_pickle(f'{path}/{train_db}/Total_train_valid/train_data_{seg_length}s.pkl')
    train_label = pd.read_pickle(f'{path}/{train_db}/Total_train_valid/train_label_{seg_length}s.pkl')
    valid_data = pd.read_pickle(f'{path}/{train_db}/Total_train_valid/valid_data_{seg_length}s.pkl')
    valid_label = pd.read_pickle(f'{path}/{train_db}/Total_train_valid/valid_label_{seg_length}s.pkl')

    lbl_count = Counter(sorted([lbl[0] for k in train_label.keys() for lbl in train_label[k]]))
    logging.info(f'Train :', dict(lbl_count))
    lbl_count = Counter(sorted([lbl[0] for k in valid_label.keys() for lbl in valid_label[k]]))
    logging.info(f'Valid :', dict(lbl_count))

    train_dataset = CustomDataset(list(train_data.values()), list(train_label.values()), seg_length=seg_length, S=S, fs=fs, C=C, B=B)
    train_loader = DataLoader(train_dataset, batch_size=BATCH, shuffle=True, num_workers=0, drop_last=True, pin_memory=True)
    valid_dataset = CustomDataset(list(valid_data.values()), list(valid_label.values()), seg_length=seg_length, S=S, fs=fs, C=C, B=B)
    validation_loader = DataLoader(valid_dataset, batch_size=BATCH, shuffle=False, num_workers=0, drop_last=True, pin_memory=True)

    model = ALPSE(in_channels=1, out_channels=out_channels, S=S, B=2, C=C, activation=activation, device=DEVICE, init_weights=True).to(DEVICE)
    loss_fn = ALPSELoss(S=S, B=B, C=C, lambda_class=lambda_class, lambda_noobj=lambda_noobj, lambda_obj=lambda_obj, lambda_iou=lambda_iou, iou_type=iou_type, reduction=reduction)

    train_loss_per_epoch = []
    valid_loss_per_epoch = []
    train_map_per_epoch = []
    valid_map_per_epoch = []

    val_min_loss = np.inf
    val_max_map = 0
    warmup_lr_final = LR

    optimizer = get_optimizer(model, optim_name, LR, MOMENTUM, WD)

    # Learning rate scheduling.
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.1)

    early_stop_count = 0
    for epoch in tqdm(range(num_epochs)):
        train_loss = []
        bl = len(train_loader)

        if epoch <= warmup_epochs:
            warmup_lr = warmup_lr_initial + (epoch / warmup_epochs) * (warmup_lr_final - warmup_lr_initial)
            for param_group in optimizer.param_groups:
                param_group['lr'] = warmup_lr

        model.train()
        for x, y in tqdm(train_loader):
            x = x.unsqueeze(1).float().to(DEVICE)
            y = y.to(DEVICE)

            lr = get_lr(optimizer) 

            optimizer.zero_grad()
            output = model(x)
            loss = loss_fn(output, y)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            
            train_loss.append(float(loss))
        train_loss_per_epoch.append(sum(train_loss)/len(train_loss))

        model.eval()
        validation_loader_idx = 0
        val_loss = []
        for x2, y2 in tqdm(validation_loader):
            with torch.no_grad():
                x2 = x2.unsqueeze(1).float().to(DEVICE)
                y2 = y2.to(DEVICE)

                output = model(x2).to(DEVICE)
                loss = loss_fn(output, y2)
                
                val_loss.append(float(loss))
        valid_loss_per_epoch.append(sum(val_loss) / len(val_loss))  

        with open(FOLDER_PATH+'/Train_Validation result.txt', 'a') as f:
            f.write(f'epoch:{epoch+1} Train loss: {np.round((sum(train_loss)/len(train_loss)), 4)} Validation loss: {np.round((sum(val_loss) / len(val_loss)), 4)} LR: {lr}\n')

        if (sum(val_loss) / len(val_loss)) < val_min_loss:
            early_stop_count = 0
            val_min_loss = (sum(val_loss) / len(val_loss))
            checkpoint = {
                            "state_dict": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "epochs": epoch+1,
                            "lr_scheduler" : lr_scheduler.state_dict(),
                        }
            torch.save(checkpoint, FOLDER_PATH+'/best_model')
        else:
            early_stop_count += 1
        
        if early_stop_count == early_stop:
            with open(FOLDER_PATH+'/Train_Validation result.txt', 'a') as f:
                f.write(f'early_stopping | epochs {epoch}')
            break

        checkpoint = {
                        "state_dict": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "epochs": epoch+1,
                        "lr_scheduler" : lr_scheduler.state_dict(),
                    }
        torch.save(checkpoint, FOLDER_PATH+'/current')

        plot_loss_per_epoch(train_losses=train_loss_per_epoch, valid_losses=valid_loss_per_epoch, FOLDER_PATH=FOLDER_PATH)

        # Evaluation
        if (epoch+1) % 10 == 0: # calculate map per 10 epochs
            model.eval()
            with torch.no_grad():
                pred_boxes, target_boxes = get_bboxes(train_loader, model, S=S, C=C, iou_threshold=iou_threshold, confidence_threshold=confidence_threshold, diou=diou, device=DEVICE)
                total_train_map, mean_precision, mean_recall, mean_f1, total_iou = mean_average_precision(pred_boxes, target_boxes, iou_threshold=iou_threshold, num_classes=C)
                train_map_per_epoch.append(total_train_map)

                pred_boxes, target_boxes = get_bboxes(validation_loader, model, S=S, C=C, iou_threshold=iou_threshold, confidence_threshold=confidence_threshold, diou=diou, device=DEVICE)
                total_valid_map, mean_precision, mean_recall, mean_f1, total_iou = mean_average_precision(pred_boxes, target_boxes, iou_threshold=iou_threshold, num_classes=C)
                valid_map_per_epoch.append(total_valid_map)
            
            if total_valid_map > val_max_map:
                val_max_map = total_valid_map
                checkpoint = {
                                "state_dict": model.state_dict(),
                                "optimizer": optimizer.state_dict(),
                                "epochs": epoch+1,
                                "lr_scheduler" : lr_scheduler.state_dict(),
                            }
                torch.save(checkpoint, FOLDER_PATH+'/maxmap')

            plot_map_per_epoch(train_maps=train_map_per_epoch, valid_maps=valid_map_per_epoch, FOLDER_PATH=FOLDER_PATH)

            with open(FOLDER_PATH+'/Train_Validation result.txt', 'a') as f:
                f.write(f'epoch:{epoch+1} Train map : {np.round(total_train_map, 4)} Validation map: {np.round(total_valid_map, 4)}\n')
            logging.info(f'epoch:{epoch+1} Train map : {np.round(total_train_map, 4)} Validation map: {np.round(total_valid_map, 4)}')
        
        lr_scheduler.step(valid_loss_per_epoch[-1]) #apply learning rate scheduler

        logging.info(f'epoch:{epoch+1} Train loss: {np.round((sum(train_loss)/len(train_loss)), 4)} Validation loss: {np.round((sum(val_loss) / len(val_loss)), 4)} LR: {lr}')


################################################ External validation ################################################
logging.info(f'External validation')
### model load
model_path = f'{FOLDER_PATH}/best_model'
model = ALPSE(in_channels=1, out_channels=out_channels, S=S, B=B, C=C, activation=activation, device=DEVICE, init_weights=True).to(DEVICE)
model_optim_state = torch.load(model_path)
model_state = model_optim_state['state_dict']
model.load_state_dict(model_state)
model.to(DEVICE) 

logging.info(f'Test data : {test_db}')

os.makedirs(f'{FOLDER_PATH}/External', exist_ok=True)
with open(f'{path}/{test_db}/data_{seg_length}s.pkl', 'rb') as f: 
    test_data = pickle.load(f)
with open(f'{path}/{test_db}/label_{seg_length}s.pkl', 'rb') as f:
    test_label = pickle.load(f)

test_dataset = CustomDataset(list(test_data[test_db].values()), list(test_label[test_db].values()), seg_length=seg_length, S=S, fs=fs, C=C, B=B)
test_loader = DataLoader(test_dataset, batch_size=BATCH, shuffle=False, num_workers=0, drop_last=False, pin_memory=True)

model.eval()
with torch.no_grad():
    pred_boxes, target_boxes = get_bboxes(test_loader, model, S=S, C=C, iou_threshold=nms_iou_threshold, confidence_threshold=confidence_threshold, diou=diou, device=DEVICE)
    total_test_map, mean_precision, mean_recall, mean_f1, total_iou = mean_average_precision(pred_boxes, copy.deepcopy(target_boxes), iou_threshold=iou_threshold, num_classes=C, savepath=f'{FOLDER_PATH}/External/{test_db}_iou05')
    
    with open(f'{FOLDER_PATH}/External/{test_db}.txt', 'a') as f:
        f.write(f'NMS threshold : {nms_iou_threshold} Confidence threshold : {confidence_threshold} DIoU nms : {diou}\n')
        f.write(f'Weight decay : {WD}\n')
        f.write(f'MAP : {np.round(total_test_map, 4)}, Precision : {np.round(mean_precision, 4)}, Recall : {np.round(mean_recall, 4)}, F1 : {np.round(mean_f1, 4)}, mIoU : {np.round(torch.mean(torch.tensor(total_iou)), 4)}\n')
    
    logging.info(f'IOU_threshold(0.5) | MAP : {total_test_map}, Precision : {mean_precision}, Recall : {mean_recall}, F1 : {mean_f1}, mIoU : {torch.mean(torch.tensor(total_iou))}')

    if test_db == 'mitdb':
        total_test_map, mean_precision, mean_recall, mean_f1, total_iou = mean_average_precision(pred_boxes, copy.deepcopy(target_boxes), iou_threshold=0.7, num_classes=C, savepath=f'{FOLDER_PATH}/External/{test_db}_iou07')

        with open(f'{FOLDER_PATH}/External/{test_db}.txt', 'a') as f:
            f.write(f'Threshold iou 0.7 | MAP : {np.round(total_test_map, 3)}, Precision : {np.round(mean_precision, 3)}, Recall : {np.round(mean_recall, 3)}, F1 : {np.round(mean_f1, 3)}, mIoU : {np.round(torch.mean(torch.tensor(total_iou)), 3)}\n')
        
        logging.info(f'IOU_threshold(0.7) | MAP : {total_test_map}, Precision : {mean_precision}, Recall : {mean_recall}, F1 : {mean_f1}, mIoU : {torch.mean(torch.tensor(total_iou))}')