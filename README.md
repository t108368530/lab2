# 實驗報告2

- [實驗報告2](#實驗報告) 
  - [說明](#說明)
  - [腳本示意圖](#腳本示意圖)
  - [腳本](#腳本)
  - [結果](#結果)



說明
---
1. 實驗環境(是使用Anaconda)
   - OS:Ubunut20.04
   - Language:Python(Version 3.8.2)
   - IDE:JupyterLab
2. 簡易步驟
   - 分切資料 -> 特徵取樣 -> 創建模型 -> 訓練模型 -> 測試模型

3. 資料集組成
  - 訓練資料= 訓練資料夾內音檔(沒調音)+訓練資料夾內音檔(Pitch Shift 0～3)
  - 驗證資料= 驗證資料夾內音檔(沒調音)
  - 測試資料= 測試資料夾內音檔


腳本示意圖
---
1. 資料切割
   ![](https://i.imgur.com/HJupvJr.png)
   
2. 特徵取樣
   ![](https://i.imgur.com/8bgkPSV.png)
   
   
腳本
---
- 複製檔案結構
```console=
cd RAW &&find . -type d | cpio -dumpl ../train &&find . -type d | cpio -dumpl ../valid &&find . -type d | cpio -dumpl ../pitchShiftTest
```
- 資料分切
```python=
def check(splitNum: float, folder: str):
    for dirname in os.listdir(folder):
        for (dirpath, dirnames, filenames) in os.walk(os.path.join(folder, dirname)):
            yield dirname, int(len(filenames) * splitNum),int(len(filenames) * splitNum* splitNum), filenames


def split(
    splitNum: float,
    rawFolder: str,
    trainFolder: str,
    validFolder: str,
    pitchShiftFolder:str,
    shuf: bool = True,
):
    """[切割資料]
    
    Arguments:
        splitNum {float} -- [訓練資料需要多少百分比,範例：60% =0.6]
        rawFolder {str} -- [原始檔案的資料夾]
        trainFolder {str} -- [訓練檔案的資料夾]
        validFolder {str} -- [驗證檔案的資料夾]
        pitchShiftFolder {str} -- [預備調音的檔案資料夾]
    
    Keyword Arguments:
        shuf {bool} -- [是否要打亂檔案] (default: {True})
    
    Returns:
        [none]] -- [沒有回傳值]
    """
    for dirname, splitnum,splitnum2 ,files in check(splitNum, rawFolder):
        if shuf:
            shuffle(files)
        else:
            files.sort()
        for file in files[:splitnum]:
            sorce = os.path.join(rawFolder, dirname, file)
            destination = os.path.join(trainFolder, dirname, file)
            copyfile(sorce, destination)
        for file in files[splitnum:splitnum2+splitnum]:
            sorce = os.path.join(rawFolder, dirname, file)
            destination = os.path.join(validFolder, dirname, file)
            copyfile(sorce, destination)
        for file in files[splitnum2+splitnum:]:
            sorce = os.path.join(rawFolder, dirname, file)
            destination = os.path.join(pitchShiftFolder, dirname, file)
            copyfile(sorce, destination)
```


- 取音頻不同特徵得的平均數
```python=
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import librosa
import random
import os
from random import random, uniform
from tqdm import tqdm


%matplotlib inline
allFilesList: list = []

dirName = ["train", "valid","test"]

trainFolders: list = sorted(os.listdir(f"./{dirName[0]}/"))
trainList: list = []

validFolders: list = sorted(os.listdir(f"./{dirName[1]}/"))
validList: list = []

pitchShiftFolders: list = sorted(os.listdir(f"./{dirName[2]}/"))
pitchShiftList: list = []
    
categorys: list = [f"{i}".rjust(2, "0") for i in range(1, 11)]
categorysCounts: dict = {"train": {}, "valid": {}, "pitch_shift": {}}

for folder in tqdm(trainFolders):
    files = os.listdir("./train/" + folder)
    for num, file in enumerate(files):
        allFilesList.append("./train/" + folder + "/" + file)
        trainList.append(folder + "/" + file)
        categorysCounts["train"][folder] = num + 1


for folder in tqdm(validFolders):
    files = os.listdir("./valid/" + folder)
    for num, file in enumerate(files):
        allFilesList.append("./valid/" + folder + "/" + file)
        validList.append(folder + "/" + file)
        categorysCounts["valid"][folder] = num + 1

for folder in tqdm(pitchShiftFolders):
    files = os.listdir(f"./{dirName[2]}/" + folder)
    for num, file in enumerate(files):
        allFilesList.append(f"./{dirName[2]}/" + folder + "/" + file)
        pitchShiftList.append(folder + "/" + file)
        categorysCounts["pitch_shift"][folder] = num + 1

trainList = sorted(trainList)
validList = sorted(validList)
pitchShiftList = sorted(pitchShiftList)
allFilesList = sorted(allFilesList)


def extractFeatures(
    path: str, ps: bool = False, ts: bool = False, st: int = 4
) -> np.ndarray:
    """[提取特徵]
    
    Arguments:
        path {str} -- [路徑]
        ps {bool} 
    Returns:
        np.ndarray -- 
               [
                mfccs,
                mfcc_delta,
                mfcc_delta2,
                chroma,
                mel,
                contrast,
                tonnetz,
                cent,
                flatness,
                rolloff,
                rms,
                ]
    """
    try:
        X, sampleRate = librosa.load(
            path, offset=0.0, res_type="kaiser_best", dtype=np.float32,
        )
        if ps:
            X = librosa.effects.pitch_shift(X, sampleRate, n_steps=st)

        mel = np.mean(librosa.feature.melspectrogram(X, sr=sampleRate).T, axis=0)
        tonnetz = np.mean(
            librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sampleRate).T,
            axis=0,
        )
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sampleRate, n_mfcc=40).T, axis=0)
        mfcc_delta = librosa.feature.delta(mfccs)  # TONY
        mfcc_delta2 = librosa.feature.delta(mfccs, order=2)  # TONY
        stft = np.abs(librosa.stft(X))
        chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sampleRate).T, axis=0)
        contrast = np.mean(
            librosa.feature.spectral_contrast(S=stft, sr=sampleRate).T, axis=0
        )
        ###### ADD NEW FEATURES (SPECTRAL RELATED)##### 24-SEP
        cent = np.mean(librosa.feature.spectral_centroid(y=X, sr=sampleRate).T, axis=0)
        flatness = np.mean(librosa.feature.spectral_flatness(y=X).T, axis=0)
        rolloff = np.mean(
            librosa.feature.spectral_rolloff(S=stft, sr=sampleRate).T, axis=0
        )
        rms = np.mean(librosa.feature.rms(S=stft).T, axis=0)
        ext_features = np.hstack(
            [
                mfccs,
                mfcc_delta,
                mfcc_delta2,
                chroma,
                mel,
                contrast,
                tonnetz,
                cent,
                flatness,
                rolloff,
                rms,
            ]
        )

    except Exception as e:
        print("Error encountered while parsing file:%s" % (path))
        return None

    return np.array(ext_features)


def creatSets(
    path: str, dataList: list, shape: int, ps: bool = False, st: float = 4
) -> (np.ndarray, np.ndarray):
    """[創建訓練資料]
    
    Arguments:
        path {str} -- [路徑]
        dataList {list} -- [檔案列表]
        shape {tuple} -- [矩陣維度]
    Returns:
        [(np.ndarray, np.ndarray)] -- [(特徵,種類)]
    """
    xArray = np.zeros([len(dataList), shape])
    yArray = np.zeros([len(dataList)])

    for index, file in tqdm(enumerate(dataList)):
        file = path + file
        try:
            xArray[index] = extractFeatures(file, ps=ps, st=st)
            yArray[index] = file.rsplit("/", 2)[1]
        except ValueError:
            print(index, file, ValueError)
    return (xArray, yArray)


def creatAugmentSets(
    path: str, dataList: list, shape: int, percent: float = 0
) -> (np.ndarray, np.ndarray):
    """[創建訓練資料]
    
    Arguments:
        path {str} -- [路徑]
        dataList {list} -- [檔案列表]
        shape {tuple} -- [矩陣維度]
    Returns:
        [(np.ndarray, np.ndarray)] -- [(特徵,種類)]
    """
    fileCounts = len(dataList)
    xArray = np.zeros([fileCounts * 5, shape])
    yArray = np.zeros([fileCounts * 5], dtype=np.int8)

    for index, file in tqdm(enumerate(dataList)):
        file = path + file
        try:
            st = uniform(1.0, 2.0)
            st2 = uniform(2.0, 3.0)
            st3 = uniform(3.0, 5.0)
            st4 = uniform(5.0, 7.0)
            ps = random() > percent

            xArray[index] = extractFeatures(file)
            yArray[index] = np.int8(file.rsplit("/", 2)[1])
            xArray[fileCounts + index] = extractFeatures(file, ps=ps, st=st)
            yArray[fileCounts + index] = np.int8(file.rsplit("/", 2)[1])
            xArray[fileCounts * 2 + index] = extractFeatures(file, ps=ps, st=st2)
            yArray[fileCounts * 2 + index] = np.int8(file.rsplit("/", 2)[1])
            xArray[fileCounts * 3 + index] = extractFeatures(file, ps=ps, st=st3)
            yArray[fileCounts * 3 + index] = np.int8(file.rsplit("/", 2)[1])
            xArray[fileCounts * 4 + index] = extractFeatures(file, ps=ps, st=st4)
            yArray[fileCounts * 4 + index] = np.int8(file.rsplit("/", 2)[1])
        except ValueError:
            print(index, file, ValueError)
    return (xArray, yArray)
```
- 對資料做常態分佈
```python=
def zScore(x):
    return (x - np.mean(x, axis=0)) / np.std(x, axis=0)
```

結果
---
- Multilayer Perceptron(MLP)
  - 訓練參數:epochs=200,batch_size=128
  - 測試結果1：
    
    ```
    model.evaluate(validData, validLabel)
    ```
    
    ```
    3/3 [==============================] - 0s 976us/step - loss: 0.6233 - accuracy: 0.8846
    [0.6233149170875549, 0.8846153616905212]
    正確率約88%
    ```
  - 測試結果2:
    ```
    model.evaluate(testData, testLabel)
    ```
    ```
    3/3 [==============================] - 0s 809us/step - loss: 0.8455 - accuracy: 0.9250
    [0.8455209136009216, 0.925000011920929]
    正確率約92%
    ```
  - 混淆矩陣：
    ![](https://i.imgur.com/Tu0zwIL.png)



- Convolutional Neural Networks(CNN)
  - 訓練參數:epochs=60,batch_size=32,
  - 測試結果1：
    
    ```
    model2.evaluate(validData.reshape(78, 277, 1), validLabel)
    ```
    
    ```
    3/3 [==============================] - 0s 2ms/step - loss: 0.6690 - accuracy: 0.8718
    [0.66900634765625, 0.8717948794364929]
    正確率約87%
    ```
  - 測試結果2:
    ```       
    model2.evaluate(testData.reshape(80, 277, 1), testLabel,batch_size=1,)
    ```
    ```
    100/100 [==============================] - 0s 116us/sample - loss: 0.4603 - accuracy: 0.9100
    [0.46028230369091033, 0.91]
    正確率約91%
    ```
  - 混淆矩陣：
    ![](https://i.imgur.com/eSuJanF.png)

