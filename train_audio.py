import os, glob, argparse
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Input, Conv1D, MaxPooling1D, LSTM, Dense, Dropout, BatchNormalization
from keras.utils import to_categorical
from preprocessing import extract_mfcc, audio_label_from_ravdess

LABELS = ["neutral","calm","happy","sad","angry","fearful","disgust","surprised"]
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--data",default="data/audio")
    args=p.parse_args()
    paths=glob.glob(os.path.join(args.data,"**","*.wav"),recursive=True)
    if not paths: raise SystemExit(f"No WAV files found in {args.data}")
    X=[]; y=[]
    for i,path in enumerate(paths,1):
        try:
            X.append(extract_mfcc(path))
            y.append(LABELS.index(audio_label_from_ravdess(os.path.basename(path))))
        except Exception as e: print("Skipping",path,e)
        if i%100==0: print(f"Processed {i}/{len(paths)}")
    X = np.asarray(X, dtype="float32")
    y=to_categorical(np.asarray(y),num_classes=len(LABELS))
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42,stratify=np.argmax(y,axis=1))
    model=Sequential([Input(shape=Xtr.shape[1:]),Conv1D(64,3,activation="relu"),BatchNormalization(),
        MaxPooling1D(2),Conv1D(128,3,activation="relu"),MaxPooling1D(2),LSTM(64),
        Dropout(.3),Dense(64,activation="relu"),Dense(8,activation="softmax")])
    model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=["accuracy"])
    model.fit(Xtr,ytr,validation_split=.1,epochs=35,batch_size=32)
    print("Test:",model.evaluate(Xte,yte,verbose=0))
    os.makedirs("models",exist_ok=True); model.save("models/audio_model.keras")
if __name__=="__main__": main()
