# Use the official Python 3.11 image
FROM python:3.11

# Set the working directory inside the server
WORKDIR /code

# Install the missing Linux graphics drivers for OpenCV
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0

# Copy the requirements file and install them
COPY ./requirements.txt /code/requirements.txt

# Force install the lightweight CPU versions of PyTorch to save server space
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# Copy all your files into the server
COPY . .

# Hugging Face requires apps to run on port 7860
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]