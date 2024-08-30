# Captioner Image Captioning Tool

This Python script processes a folder of images, renames them, and generates captions using a language model. It's designed to work with various image formats and can be customized to focus on specific subjects or artistic styles.

## Features

- Processes multiple image formats (png, jpg, jpeg, gif, bmp, tiff, webp)
- Renames images with a custom prefix and sequential numbering
- Generates captions using a language model (default: gpt-4o-mini)
- Supports two caption modes: subject-focused and style-focused
- Concurrent processing for improved performance

## Requirements

- Python 3.6+
- Required Python packages are listed in `requirements.txt`

## Setup

1. Clone this repository or download the script.
2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```
3. Create a `.env` file in the same directory as the script and add your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

## Usage

Run the script from the command line with the following syntax:

```
python captioner.py <folder_path> --prefix <prefix> --prefix_type <subject|style>
```

Arguments:

- `folder_path`: Path to the folder containing the images (required)
- `--prefix`: Prefix to prepend to the renamed image and text files (optional)
- `--prefix_type`: Type of prefix, either "subject" or "style" (required)

Example:

```
python captioner.py ./my_images --prefix "cat_" --prefix_type subject
```

This command will process all images in the `./my_images` folder, rename them with the prefix "cat\_", and generate captions focusing on the subject (in this case, cats).

## Output

The script will:

1. Rename all images in the specified folder to `<prefix><number>.jpg`
2. Generate a caption for each image
3. Save each caption in a text file named `<prefix><number>.txt`
4. Display progress and results in the console

## Customization

You can modify the `make_langchain_call` function to adjust the prompts or change the language model used for captioning.

## Note

Ensure you have the necessary permissions to read from and write to the specified folder. The script will overwrite existing files with the same names, so use caution when specifying the prefix.
