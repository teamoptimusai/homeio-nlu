FROM pytorch/pytorch:latest

WORKDIR /nlu
COPY . /nlu
RUN pip install -r requirements.txt
# RUN sh scripts/get_pretrain.sh

EXPOSE 5000

CMD ["python", "engine.py","--weights","scripts/epoch50_best_model_trace.pth"]