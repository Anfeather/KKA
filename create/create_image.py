from zhipuai import ZhipuAI
import os
import requests

os.environ["http_proxy"] = "http://localhost:7890"
os.environ["https_proxy"] = "http://localhost:7890"


client = ZhipuAI(api_key="your APIKey") 

# Read the description from the txt file
def read_descriptions(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    descriptions = [sentence.strip() for sentence in content.split('\n') if sentence.strip()]
    return descriptions


# Save picture to local
def save_image(image_url, save_path):
    response = requests.get(image_url)
    if response.status_code == 200:
        with open(save_path, 'wb') as file:
            file.write(response.content)
        print(f"Image saved to {save_path}")
    else:
        print(f"Failed to retrieve image from {image_url}")


def main():
    file_path = 'flower.txt'  # Please fill in your txt file path
    descriptions = read_descriptions(file_path)

    for idx, description in enumerate(descriptions):
        print(f"Processing description {idx + 1}/{len(descriptions)}: {description}")
        try:
            response = client.images.generations(
                model="cogview-3-plus",  # Fill in the name of the model you want to call
                prompt=description,
            )

            image_url = response.data[0].url

            # save images
            save_path = f"image_{idx + 1}.jpg"
            save_image(image_url, save_path)
        except Exception as e:
            print(f"Error processing description {idx + 1}: {e}")
if __name__ == "__main__":
    main()
