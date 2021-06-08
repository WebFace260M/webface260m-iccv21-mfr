# WebFace260M Track of ICCV21-MFR
The Masked Face Recognition Challenge & Workshop(MFR) will be held in conjunction with the International Conference on Computer Vision (ICCV) 2021.

[Challenge-website](https://www.face-benchmark.org/challenge.html).

[Workshop-Homepage](https://ibug.doc.ic.ac.uk/resources/masked-face-recognition-challenge-workshop-iccv-21/).

There're [InsightFace track](https://github.com/deepinsight/insightface/tree/master/challenges/iccv21-mfr) and Webface260M track(with larger training set) here in this workshop.

### Inference demo
1. Clone the repository. We call the directory ``webface260m-iccv21-mfr`` as *`MFR_ROOT`*.
```Shell
git clone https://github.com/WebFace260M/webface260m-iccv21-mfr.git
```
2. Download the model files from [BaiduYun](https://pan.baidu.com/s/1Zd62dC0rVBLlc2Drspi0ow)[extraction code: ``5pig``] or [Dropbox] and unzip.
```Shell
cd $MFR_ROOT/demo
unzip assets.zip
```
4. Pull the docker file and open it
```Shell
docker pull webface260m/mfr:latest
docker run -it -v /mnt/:/mnt/ --name mfr webface260m/mfr /bin/bash
```
3. (Optional) If you do not run step 2, you can also run the following command to install the necessary packages.
```Shell
cd $MFR_ROOT/demo
pip install -r requirements.txt
```
5. Run the demo code
```Shell
cd $MFR_ROOT/demo
python demo_feat.py
```
In this demo, we provide a face detection model (``$MFR_ROOT/assets/det/R50-retinaface.onnx``) and a face recognition model (``$MFR_ROOT/assets/face_reg/R18.onnx``). Participants should replace them with the models trained by themselves for submission to [codalab](https://competitions.codalab.org/competitions/32478).

### Submission Guide
1. Participants should put all models and files into ``$MFR_ROOT/assets/``.
2. Participants must provide ``$MFR_ROOT/pywebface260mmfr_implement.py`` which contains the ``PyWebFace260M`` class.  
3. Participants should run the ``demo_feat.py`` in ``$MFR_ROOT/demo/``  on the provided docker file to ensure the correctness of feature and time constraints.  
4. Participants must package the code directory for submission using ``zip -r xxx.zip $MFR_ROOT`` and then upload it to [codalab](https://competitions.codalab.org/competitions/32478).  
5. Please sign-up with the real organization name. You can hide the organization name in our system if you like.  
6. You can decide which submission to be displayed on the leaderboard.
