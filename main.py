import os
import base64
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain.prompts import ChatPromptTemplate 
from langchain_openai import ChatOpenAI
import argparse
import concurrent.futures
import tqdm
import re

def is_image_file(filename: str):
    image_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp')
    return filename.lower().endswith(image_extensions)

def load_image_as_base64(image_path: str):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def make_langchain_call(image_path: str, prefix: str, prefix_type: str, mode: str, model_name: str ="gpt-4-vision-preview"):
    image_data = load_image_as_base64(image_path)

    message = HumanMessage(
            content=[
            {"type": "text", "text": "Image Captioning Task:"},
                {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
                },
            ],
        )

    system_prompt = (
        "You are a professional image captioner. You are given an image and will provide a caption for it. "
        "You should always provide a caption that is relevant to the image. "
        "This caption should capture aspects of the subject and setting of the image. "
        f"Please make sure the given prefix: {prefix} is included in the caption naturally. "
        f"The prefix indicates a {prefix_type}. "
    )

    if prefix_type == "subject":
        system_prompt += (
            f"Focus on describing the {prefix} in detail. "
            "Include things such as expressions, poses, camera angles, lighting, surroundings, distinctive features, etc. "
            f"Example: A photo of {prefix}. A playful golden retriever frolicking in a sunlit park. "
        )
    elif prefix_type == "style":
        system_prompt += (
            f"Ensure the caption reflects the artistic style of {prefix}. "
            f"Example: An image in the style of {prefix}. A vibrant anime-style illustration of a determined schoolgirl with large expressive eyes and colorful hair. "
        )

    system_prompt += (
        "Balance: Strive for a balance between detail and conciseness. The captions should be detailed enough to capture the essential characteristics and distinctions in the images, but not so lengthy that they introduce unnecessary complexity. "
    )

    if mode == "file_name":
        system_prompt += (
            "IMPORTANT: Your caption will be used as a file name. It MUST be 256 characters or less, including spaces. "
            "Do not use any characters that are invalid in file names (/, \\, :, *, ?, \", <, >, |). "
            "Make sure the caption is concise but still descriptive and includes the prefix."
        )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            (message)
        ]
    )

    model = ChatOpenAI(model=model_name)
    response = model.invoke(prompt.format_messages())
    return response.content

def sanitize_filename(filename: str):
    # Remove invalid characters
    filename = re.sub(r'[<>:"/\\|?*]', '', filename)
    # Replace spaces with underscores
    filename = filename.replace(' ', '_')
    # Truncate to 256 characters
    return filename[:256]

def process_image(args):
    folder_path, filename, prefix, prefix_type, index, mode = args
    file_path = os.path.join(folder_path, filename)
    file_extension = os.path.splitext(filename)[1]

    response = make_langchain_call(file_path, prefix, prefix_type, mode)

    if mode == "file_name":
        new_filename = sanitize_filename(response) + file_extension
        new_file_path = os.path.join(folder_path, new_filename)
        os.rename(file_path, new_file_path)
        return new_filename, None
    else:
        new_filename = f"{prefix}{index:02}{file_extension}"
        new_file_path = os.path.join(folder_path, new_filename)
        os.rename(file_path, new_file_path)
        response_file = os.path.join(folder_path, f"{prefix}{index:02}.txt")
        with open(response_file, "w") as f:
            f.write(str(response))
        return new_filename, response_file

def loop_through_images_and_call_langchain(folder_path: str, prefix: str, prefix_type: str, mode: str):
    if not os.path.isdir(folder_path):
        print(f"The provided path '{folder_path}' is not a directory or does not exist.")
        return

    image_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and is_image_file(f)]
    image_files.sort()

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for index, filename in enumerate(image_files, start=1):
            futures.append(executor.submit(process_image, (folder_path, filename, prefix, prefix_type, index, mode)))

        for future in tqdm.tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing images"):
            new_filename, response_file = future.result()
            print(f"Processed image: {new_filename}")
            if response_file:
                print(f"Response saved to: {response_file}")

def main():
    load_dotenv()
    
    parser = argparse.ArgumentParser(description="Loop through all images in a folder and send them to a language model.")
    parser.add_argument("folder_path", type=str, help="Path to the folder containing images.")
    parser.add_argument("--prefix", type=str, default="", help="Prefix to prepend to the renamed image and text files.")
    parser.add_argument("--prefix_type", type=str, choices=["subject", "style"], required=True, help="Type of prefix: 'subject' or 'style'")
    parser.add_argument("--mode", type=str, choices=["text_file", "file_name"], default="text_file", help="Mode of operation: 'text_file' or 'file_name'")
    
    args = parser.parse_args()
    
    loop_through_images_and_call_langchain(args.folder_path, args.prefix, args.prefix_type, args.mode)

if __name__ == "__main__":
    main()
