FROM python:3.10
WORKDIR /oit_matches
COPY ./ /oit_matches
RUN pip install -r /oit_matches/requirements.txt --no-cache-dir
EXPOSE 8000
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]