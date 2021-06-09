# WebFace260M Track of ICCV21-MFR
The Masked Face Recognition Challenge & Workshop(MFR) will be held in conjunction with the International Conference on Computer Vision (ICCV) 2021.

[Challenge-website](https://www.face-benchmark.org/challenge.html).

[Workshop-Homepage](https://ibug.doc.ic.ac.uk/resources/masked-face-recognition-challenge-workshop-iccv-21/).

Submission server link: https://competitions.codalab.org/competitions/32478

There're [InsightFace track](https://github.com/deepinsight/insightface/tree/master/challenges/iccv21-mfr) and Webface260M track(with larger training set) here in this workshop.

## Recent Update
**`2021-06-09`**: We provided a version which partly fixed the problem of ``input_mean`` and ``input_std``.

**`2021-06-08`**: We released the submission package for [WebFace260M Track of ICCV21-MFR](https://www.face-benchmark.org/challenge.html).


### Inference demo
1. Clone the repository. We call the directory ``webface260m-iccv21-mfr`` as *`MFR_ROOT`*.
```Shell
git clone https://github.com/WebFace260M/webface260m-iccv21-mfr.git
```
2. Download the model files from [BaiduYun](https://pan.baidu.com/s/1Zd62dC0rVBLlc2Drspi0ow)[extraction code: ``5pig``] or [Dropbox](https://www.dropbox.com/s/cw52tmxgu1cboii/assets.zip?dl=0) and unzip.
```Shell
cd $MFR_ROOT/
unzip assets.zip
```
3. Pull the docker file and open it
```Shell
docker pull webface260m/mfr:latest
docker run -it -v /mnt/:/mnt/ --name mfr webface260m/mfr /bin/bash
```
4. (Optional) If you do not run ``step 3``, you can also run the following command to install the necessary packages.
```Shell
cd $MFR_ROOT
pip install -r requirements.txt
```
5. Run the demo code
```Shell
cd $MFR_ROOT/demo
python demo_feat.py
```
### Important Notes
1. In the above demo, we provide a face detection model (``$MFR_ROOT/assets/det/R50-retinaface.onnx``) and a face recognition model (``$MFR_ROOT/assets/face_reg/R18.onnx``). Participants should replace them with the models trained by themselves and modify the model name in [``pywebface260mmfr_implement.py``](https://github.com/WebFace260M/webface260m-iccv21-mfr/blob/main/pywebface260mmfr_implement.py)
```Shell
    def load(self, rdir):
        det_model = os.path.join(rdir, 'det', 'R50-retinaface.onnx')
        self.detector = face_detection.get_retinaface(det_model)
        print('use onnx-model:', det_model)

        self.detector.prepare(use_gpu=self.use_gpu)
        max_time_cost = 1000

        self.model_file = os.path.join(rdir, 'face_reg', "R18.onnx")
        print('use onnx-model:', self.model_file)
```
accordingly, for submission to [codalab](https://competitions.codalab.org/competitions/32478).

2. Participants must judge whether the value of ``input_mean`` and ``input_std`` in [``pywebface260mmfr_implement.py``](https://github.com/WebFace260M/webface260m-iccv21-mfr/blob/main/pywebface260mmfr_implement.py) is right for their own face recognition models. Generally, when the model have preprocessing steps in itself, it do not need any other operations (``input_mean = 0.0`` and ``input_std = 1.0``). Otherwise, it may be ``input_mean = 127.5`` and ``input_std = 127.5``. Participants must adjust the script accordingly.
3. After replacing models and before submission, participants must compare the feature obtained by this repo with their own framework (``mxnet, pytorch``) by using one image, e.g. ``demo/0.png``, to make sure the feature are almost the same. Otherwise, there may be other issues.
4. Participants must ensure 1000 ms constrain for whole face recognition system. At present, the submission exceeding 2000ms will not be evaluated.

### Submission Guide
1. Participants should put all models and files into ``$MFR_ROOT/assets/``.
2. Participants must provide ``$MFR_ROOT/pywebface260mmfr_implement.py`` which contains the ``PyWebFace260M`` class.  
3. Participants should run the ``demo_feat.py`` in ``$MFR_ROOT/demo/``  on the provided docker file to ensure the correctness of feature and time constraints.  
4. Participants must package the code directory for submission using ``zip -r xxx.zip $MFR_ROOT`` and then upload it to [codalab](https://competitions.codalab.org/competitions/32478).  An example directory after unzipping the submission zip file:
```Shell
[mfr_submit_code]$ tree
.
├── assets
│   ├── det
│   │   └── R50-retinaface.onnx
│   └── face_reg
│       └── R18.onnx
├── face_detection.py
└── pywebface260mmfr_implement.py

3 directories, 4 files
```
5. Please sign-up with the real organization name. You can hide the organization name in our system if you like.  
6. You can decide which submission to be displayed on the leaderboard.
