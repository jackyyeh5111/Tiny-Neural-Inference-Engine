from PIL import Image
from torchvision import transforms


def export_bytes(image_path):
    transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    img = Image.open(image_path)
    img_tensor = transform(img).unsqueeze(0)  # Shape: (1, 1, 28, 28)

    # Convert to numpy and ensure it is float32 (matches C++ float)
    # .numpy() sharing memory, .flatten() makes it a 1D array of 784 elements
    data = img_tensor.detach().cpu().numpy().astype("float32").flatten()

    # Write raw bytes to file
    image_name = image_path.split("/")[-1].split(".")[0]
    out_dir = "/".join(image_path.split("/")[:-1])
    out_file = f"{out_dir}/{image_name}.bin"
    with open(out_file, "wb") as f:
        f.write(data.tobytes())

    print(f"Tensor exported to {out_file} ({len(data)} floats)")


if __name__ == "__main__":
    # Usage
    image_file = "./images/num_2.jpg"

    # export
    export_bytes(image_file)
