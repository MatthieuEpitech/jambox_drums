FROM ubuntu:18.04

ENV TZ=Europe/Paris

RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN apt-get update && apt-get install -y python3.7 python3-pip && apt-get install -qq libfluidsynth1 fluid-soundfont-gm build-essential libasound2-dev libjack-dev

COPY requirements.txt .

RUN python3 -m pip install --upgrade pip
RUN pip3 install -r requirements.txt

COPY *.py ./
COPY data/ ./data/
COPY models/ ./models/

ENV BASE_FILENAME='loop.wav'
ENV PATH_MODEL='./models/'
ENV PATH_DATA='./data/'

CMD ["python3", "main.py"]