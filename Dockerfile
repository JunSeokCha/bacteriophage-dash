FROM python:3.10.5
COPY ./ ./
RUN pip3 install -r requirements.txt
WORKDIR /
EXPOSE 8050
CMD ['python3', './run.py']

