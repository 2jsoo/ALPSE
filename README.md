# ALPSE : Arrhythmia Localization using Parameter-efficient deep learning model from Single-lead ECG

**[Model]**
Architecture.png

## Datasets
- Long Term AF Database (LTAFDB) : 84 recordings from 84 patients, 128 Hz, 24-25 hours per recording
- MIT-BIH Atrial Fibrillation Database (AFDB) : 23 recordings from 23 patients, 256 Hz, 10 hours per recording
- The 4th China Physiological Signal Challenge 2021 (CPSC2021) : 1,436 recordings from 105 patients, 200 Hz, 20 minutes per recording
- MIT-BIH Arrhythmia Database (MITDB) : 48 recordings from 47 patients, 360 Hz, 30 minutes per recording
- MIT-BIH Normal Sinus Rhythm Database (NSRDB) : 18 recordings from 18 patients, 128 Hz, over 24 hours per recording

## Running Guide
```bash
python train.py --
python test.py --
```

## Performance
