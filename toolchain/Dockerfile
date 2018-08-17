# should be upload as leelabcnbc/2018-attention-project (no tag)
FROM leelabcnbc/leelab-py3:20180816

# install scikit-image, scikit-learn, joblib
#
RUN conda install -c conda-forge -n leelab -y --no-update-dependencies \
   scikit-image=0.14.0 scikit-learn=0.19.2 joblib=0.12.2 \
   && conda clean --all

# install pytorch, pip style, CUDA 9.0
RUN /opt/conda/envs/leelab/bin/pip install --no-cache-dir \
    http://download.pytorch.org/whl/cu90/torch-0.4.1-cp36-cp36m-linux_x86_64.whl \
    torchvision
