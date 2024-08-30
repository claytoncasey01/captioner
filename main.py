import os
import base64
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain.prompts import ChatPromptTemplate 
from langchain_openai import ChatOpenAI
import argparse

def is_image_file(filename: str):
    image_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp')
    return filename.lower().endswith(image_extensions)

def load_image_as_base64(image_path: str):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def make_langchain_call(image_path: str, prefix: str, model_name: str ="gpt-4o-mini"):
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
        f"Please make sure the given prefix is included in the caption."
        "Balance: Strive for a balance between detail and conciseness. The captions should be detailed enough to capture the essential characteristics and distinctions in the images, but not so lengthy that they introduce unnecessary complexity. "
        "Include things such as expressions, poses, camera angles, lighting, surroundings, distinctive features, etc. "
        "Example - Image: A photo of a dog in a park. Caption: A dog playing in the park."
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

def loop_through_images_and_call_langchain(folder_path: str, prefix: str):
    if not os.path.isdir(folder_path):
        print(f"The provided path '{folder_path}' is not a directory or does not exist.")
        return

    image_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and is_image_file(f)]
    image_files.sort()

    for index, filename in enumerate(image_files, start=1):
        file_path = os.path.join(folder_path, filename)
        new_filename = f"{prefix}{index:02}.jpg"  # Renames images as 01.jpg, 02.jpg, etc.
        new_file_path = os.path.join(folder_path, new_filename)

        os.rename(file_path, new_file_path)  # Renames the image file

        print(f"Processing image: {new_filename}")
        response = make_langchain_call(new_file_path, prefix)

        response_file = os.path.join(folder_path, f"{prefix}{index:02}.txt")
        with open(response_file, "w") as f:
            f.write(str(response))

        print(f"Response saved to: {response_file}")

def main():
    load_dotenv()
    
    parser = argparse.ArgumentParser(description="Loop through all images in a folder and send them to a language model.")
    parser.add_argument("folder_path", type=str, help="Path to the folder containing images.")
    parser.add_argument("--prefix", type=str, default="", help="Prefix to prepend to the renamed image and text files.")
    
    args = parser.parse_args()
    
    loop_through_images_and_call_langchain(args.folder_path, args.prefix)

if __name__ == "__main__":
    main()
