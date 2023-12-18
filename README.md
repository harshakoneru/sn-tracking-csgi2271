# SoccerNet MOT Challenge 

**Acknowledgements**

Significant portions of this codebase has been adapted from the BytrTrack and Deep OC-SORT github pages. We thank the authors of those pages.

https://github.com/GerardMaggiolino/Deep-OC-SORT/

https://github.com/ifzhang/ByteTrack

**Setting Up the Environment**

```bash
conda create -n tracking python=3.8
conda activate tracking

git clone https://github.com/harshakoneru/sn-tracking-csgi2271.git
cd sn-tracking-csgi2271

pip3 install -r requirements.txt
python3 setup.py develop

pip3 install cython; pip3 install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'

cd deep-oc-sort

cd external/yolox
python setup.py develop

cd ../deep-person-reid/
pip install -r requirements.txt && python setup.py develop

cd ../fast_reid/
pip install -r docs/requirements.txt
```

**Training**

```bash
python3 tools/train.py -f exps/yolox_x_10Ep_bytetrack_mot_17_run2.py -d 2 -b 16 --fp16 -o -c /scratch/hk3820/csgi2271_finalproject/YOLOX_outputs/yolox_x_10Ep_bytetrack_mot_17/latest_ckpt.pth.tar
```

**Tracking**
