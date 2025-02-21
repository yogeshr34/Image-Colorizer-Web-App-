
# Image Colorizer Web App 
I and my teammate LathikaGS developed this ML based web application for colorization of black and white images into colourized images using autoencoders. 

This project is a web application that colorizes grayscale images using a machine learning model built with PyTorch, Torchvision, and autoencoders. The application is hosted using Streamlit, providing an intuitive interface for users to upload grayscale images and view their colorized versions.

## Features
- **Automatic Image Colorization**: Upload a grayscale image, and the app will generate a colorized version.
- **Streamlit Interface**: User-friendly web interface for seamless image uploading and colorization.
- **Autoencoder-Based Model**: The model was trained with autoencoders, specifically designed to learn and apply colors to grayscale images.

## Getting Started

### Prerequisites
- Python 3.7 or higher
- Streamlit
- PyTorch
- Torchvision

### Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/image-colorizer
   cd image-colorizer
   ```

2. **Install Dependencies**
   Use `pip` to install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download Model**
   Download the trained model file and place it in the `models/` directory. (Specify the link or instructions if necessary.)

### Usage

Run the application with the following command:
```bash
streamlit run app.py
```

Once running, you can access the app in your browser at `http://localhost:8501`.

## How It Works
The app utilizes a pre-trained autoencoder model that has been trained to map grayscale images to their color counterparts. Here’s a quick rundown of the workflow:
1. **Input**: The user uploads a grayscale image.
2. **Processing**: The model processes the image, generating a colorized version.
3. **Output**: The colorized image is displayed alongside the original grayscale image.

## Demo

Here's an example of how the colorization process transforms a grayscale image into a colorized version:

| Grayscale Image | Colorized Output |
|-----------------|------------------|
| ![Grayscale](example_grayscale.jpg) | ![Colorized](example_colorized.jpg) |

## Model Details
- **Framework**: PyTorch with Torchvision
- **Architecture**: Autoencoder with custom enhancements for colorization
- **Dataset**: The model was trained on Kaggle based dataset.

## Future Improvements
- **Enhance Model Performance**: Experiment with other architectures like GANs for better colorization.
- **Add Download Option**: Allow users to download the colorized image.
- **Model Fine-Tuning**: Allow users to adjust settings for customized results.

## Contributing
Contributions are welcome! Feel free to submit a pull request or open an issue for suggestions and bug reports.

