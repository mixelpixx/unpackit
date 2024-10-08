# Packing List Processor

This is a PDF-specific tool designed for processing packing lists. It's intended for personal use and a limited audience.

## Prerequisites


## Installation

1. Clone this repository or download the source code.

2. Navigate to the project directory:
   ```
   cd path/to/unpackit
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

## Usage

1. Create a new project or load an existing one.
2. Process PDF files containing packing lists.
3. Use the barcode scanner to check items.
4. Generate and send status reports via email.

For more detailed instructions, please refer to the in-app help or contact the developer.

## Notes

- This tool is designed for a specific workflow and may require customization for different use cases.
- Ensure your camera is properly connected and configured for the barcode scanning feature to work.
