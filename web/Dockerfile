FROM python:3

WORKDIR /app

COPY . /app

RUN apt update && \
    apt install -y locales && \
    echo "ja_JP.UTF-8 UTF-8" >> /etc/locale.gen && \
    locale-gen

RUN pip install --trusted-host pypi.python.org -r requirements.txt

EXPOSE 80

CMD ["gunicorn", "-b", "0.0.0.0:80", "web:app"]