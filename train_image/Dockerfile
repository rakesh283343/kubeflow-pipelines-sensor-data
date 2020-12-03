FROM gcr.io/mwpmltr/rrusson_kubeflow_base:v1
RUN pip install -U tensorflow==2.1.1
WORKDIR /app
COPY train.py .

ENTRYPOINT ["python", "train.py"]
