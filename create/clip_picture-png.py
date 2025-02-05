from PIL import Image
import os

# crop image
def crop_image(input_path, output_path, crop_fraction=1/13):
    with Image.open(input_path) as img:
        width, height = img.size
        crop_height = int(height * crop_fraction)
        cropped_img = img.crop((0, 0, width, height - crop_height))
        cropped_img.save(output_path)
        print(f"Cropped image saved to {output_path}")

def main():
    input_directory = 'flowers_pictures_before'   
    output_directory = 'flowers_pictures_after'  

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)


    for filename in os.listdir(input_directory):
        if filename.endswith(".png") or filename.endswith(".jpg"):
            input_path = os.path.join(input_directory, filename)
            output_path = os.path.join(output_directory, filename)
            crop_image(input_path, output_path)

if __name__ == "__main__":
    main()
