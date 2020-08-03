# IM - Integrated Model for LSST AOS
We are expanding the capabilities of the original [LSST IM code](https://github
.com/bxin/IM) by Bo Xin and explained in [arxiv:1506.04839](https://arxiv.org/pdf/1506.04839
.pdf). This will allow us to develop new algorithms and perfect strategies for the LSST Active 
Optics System (AOS).

```
CODE_DIR=/home/dthomas/Code
export LSST_DIR=$CODE_DIR/../lsst16
source $LSST_DIR/loadLSST.bash
setup obs_lsst -t sims_w_2019_08
setup -k -r $CODE_DIR/phosim_utils
PYTHONPATH=$PYTHONPATH:/home/dthomas/Code/cwfs/python:/home/dthomas/Code/
```