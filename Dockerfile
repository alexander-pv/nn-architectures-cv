
FROM nvidia/cuda:10.2-devel-ubuntu18.04

WORKDIR /home
COPY ./notebooks /home/notebooks
COPY ./assets /home/assets
COPY requirements.txt .

RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-get update && apt-get  install --yes python3.8 python3.8-dev python3-pip
RUN rm -rf /usr/share/doc/* /usr/share/man/* /var/lib/apt/lists/* /tmp/* /var/tmp/* && \
    apt-get  autoremove

RUN python3 -m pip install -U pip
RUN pip install -r requirements.txt

EXPOSE 8000

ENTRYPOINT [ "/bin/bash", "-l", "-c"]
CMD ["jupyter-notebook --allow-root --port=8000"]
