FROM python:3.10
WORKDIR /oit_matches
COPY ./ /oit_matches
RUN apk update && pip install -r /oit_matches/requirments.txt --no-cache-dir
EXPOSE 8000
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]