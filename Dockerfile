FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files
COPY data /app/data
COPY scripts /app/scripts
COPY prompts /app/prompts
COPY src /app/src

# Command to run the script
CMD ["python", "scripts/report_results.py"]
