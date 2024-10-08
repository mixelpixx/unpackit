# Packing List Processor

This is a PDF-specific tool designed for processing packing lists. It's intended for personal use and a limited audience.

## Prerequisites


## Installation

1. Clone this repository or download the source code.

2. Navigate to the project directory:
   ```
   cd path/to/packing-list-processor
   ```

3. (Optional but recommended) Create and activate a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use: venv\Scripts\activate
   ```

4. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Configuration

1. Open the `config.ini` file and update the email configuration with your details:
   ```
   [Email]
   sender_email = your-email@example.com
   password = your-password
   smtp_server = smtp.gmail.com
   smtp_port = 587
   ```

## Running the Program

To start the Packing List Processor:
