# Description of folders containing codes

This repository contains two folders named CLF, Down_sampl_Seg, HLF, LLF, Pre_proc, Sel_Ext, TD_FD_TFD, and VIS each containing MATLAB files.

## CLF Folder

This folder contains MATLAB files related to classification. 

### Files:

1. `CV_par.m`: [cross-validation partition]
2. `RF.m`: [Random forest classifier]
3. `SVM_lin_rbe.m`: [SVM classifier with linear and RBF kernel]

## Down_sampl_Seg Folder

This folder contains MATLAB files related to data segmentation and down sampling. 

### Files:

1. `CWRU_datasegmentation.m`: [Data segmentation]
2. `main_downsamplingandByAveraging.m`: [Down sampling]

## LLF Folder

This folder contains MATLAB files related to extraction low level features. 

### Files:

1. `main_CWRU_emdfeature1.m`: [This is the main program for extracting emd based low level features]

   
## HLF Folder

This folder contains MATLAB files related to extraction high level features.

### Files:

1. `Main_Program.m`: [This is the main program for High level feature extraction]

## Pre_Proc Folder

This folder contains MATLAB files related to preprocessing.

### Files:

1. `main_preprocess.m`: [This is the main program for preprocessing]
2. `fun_preprocessing.m`: [Functions for preprocessing]

## Sel_Ext Folder

This folder contains MATLAB files related to feature extraction, selection, and fusion.

### Files:

1. `main_selection.m`: [This is the main program for featureselectin/extraction]

## TD_FD_TFD Folder

This folder contains MATLAB files related to feature extraction in time, frequency, and time frequency domains.

### Files:

1. `CWRU_main_feature_extract.m`: [This is the main program for feature extraction in TD FD and TFD]

## VIS Folder

This folder contains MATLAB files related to visualization of features using tSNE.

### Files:

1. `maintSNE.m`: [This is the main program for visualization of features using tSNE]

Note: To run the programs or implementation of particular components, users need to first download the dataset from the reference given in the thesis and then organize the paths for program folders, files, and functions accordingly.
