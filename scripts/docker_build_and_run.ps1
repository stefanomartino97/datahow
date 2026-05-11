docker build -t datahow-api "$PSScriptRoot\.."
docker run -d -p 8000:8000 --name datahow datahow-api