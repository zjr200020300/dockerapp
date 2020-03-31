
FROM tensorflow/tensorflow:2.0.0-py3-jupyter
WORKDIR /
COPY dockerapp.py \
 requirements.txt \
  ./
RUN pip install -r requirements.txt
EXPOSE 8080
CMD ["python3", "/dockerapp.py"]
