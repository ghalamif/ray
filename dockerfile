FROM rayproject/ray-ml:nightly-py311-gpu

WORKDIR /app

COPY model.py train.py dataset.py testcluster.py /app/

# Clean up pip cache
RUN rm -rf ~/.cache/pip
