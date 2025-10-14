import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import vgg19
from tensorflow.keras.preprocessing.image import load_img, img_to_array, save_img
import matplotlib.pyplot as plt
import matplotlib
# Use a non-interactive backend to avoid display issues
matplotlib.use('Agg')  # This line is important for non-GUI environments
import time
import os

# Suppress TensorFlow logging
tf.get_logger().setLevel('ERROR')

class NeuralStyleTransfer:
    def __init__(self, content_path, style_path, result_prefix='generated'):
        """
        Initialize the Neural Style Transfer model.
        
        Args:
            content_path (str): Path to the content image
            style_path (str): Path to the style image
            result_prefix (str): Prefix for saving generated images
        """
        self.content_path = content_path
        self.style_path = style_path
        self.result_prefix = result_prefix
        
        # Dimensions of the generated picture
        self.img_nrows = 400
        self.img_ncols = int(self.img_nrows * 1.5)
        
        # Weights of the different loss components
        self.total_variation_weight = 1e-6
        self.style_weight = 1e-6
        self.content_weight = 2.5e-8
        
        # Layers used for style and content extraction
        self.content_layers = ['block5_conv2']
        self.style_layers = [
            'block1_conv1',
            'block2_conv1',
            'block3_conv1',
            'block4_conv1',
            'block5_conv1',
        ]
        
        # Load and preprocess images
        self.content_image = self.preprocess_image(content_path)
        self.style_image = self.preprocess_image(style_path)
        self.generated_image = tf.Variable(self.content_image)
        
        # Build the model
        self.model = self.build_model()
        
    def preprocess_image(self, image_path):
        """Load and preprocess an image for VGG19."""
        # Load and resize image
        img = load_img(image_path, target_size=(self.img_nrows, self.img_ncols))
        # Convert to array and add batch dimension
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)
        # Preprocess for VGG19
        img = vgg19.preprocess_input(img)
        return tf.convert_to_tensor(img, dtype=tf.float32)
    
    def deprocess_image(self, x):
        """Convert a tensor to a valid image."""
        x = x.reshape((self.img_nrows, self.img_ncols, 3))
        # Remove zero-center by mean pixel
        x[:, :, 0] += 103.939
        x[:, :, 1] += 116.779
        x[:, :, 2] += 123.68
        # 'BGR'->'RGB'
        x = x[:, :, ::-1]
        x = np.clip(x, 0, 255).astype('uint8')
        return x
    
    def build_model(self):
        """Build the VGG19 model with access to intermediate layers."""
        # Load VGG19 model pre-trained on ImageNet
        vgg = vgg19.VGG19(weights='imagenet', include_top=False)
        vgg.trainable = False
        
        # Get the symbolic outputs of each "key" layer
        outputs_dict = dict([(layer.name, layer.output) for layer in [vgg.get_layer(name) for name in (self.style_layers + self.content_layers)]])
        
        # Create a model that returns the layer outputs
        return tf.keras.Model(vgg.input, outputs_dict)
    
    def content_loss(self, base_content, target):
        """Calculate the content loss."""
        return tf.reduce_mean(tf.square(base_content - target))
    
    def gram_matrix(self, x):
        """Calculate the Gram matrix."""
        # Ensure we're working with a 3D tensor [height, width, channels]
        if len(x.shape) == 4:  # If batch dimension exists
            x = tf.squeeze(x, axis=0)  # Remove batch dimension
            
        # Reshape to 2D: (channels, height * width)
        features = tf.reshape(x, (tf.shape(x)[2], -1))
        
        # Calculate Gram matrix
        gram = tf.matmul(features, features, transpose_b=True)
        
        # Normalize by number of elements
        return gram / tf.cast(tf.size(x), tf.float32)
    
    def style_loss(self, style, combination):
        """Calculate the style loss."""
        S = self.gram_matrix(style)
        C = self.gram_matrix(combination)
        channels = 3
        size = self.img_nrows * self.img_ncols
        return tf.reduce_sum(tf.square(S - C)) / (4.0 * (channels ** 2) * (size ** 2))
    
    def total_variation_loss(self, x):
        """Calculate the total variation loss (smoothness prior)."""
        a = tf.square(
            x[:, : self.img_nrows - 1, : self.img_ncols - 1, :] - x[:, 1:, : self.img_ncols - 1, :]
        )
        b = tf.square(
            x[:, : self.img_nrows - 1, : self.img_ncols - 1, :] - x[:, : self.img_nrows - 1, 1:, :]
        )
        return tf.reduce_sum(tf.pow(a + b, 1.25))
    
    def compute_loss(self, generated_outputs):
        """Compute the total loss."""
        # Extract features from the generated image
        generated_features = self.model(self.generated_image)
        
        # Initialize loss
        loss = tf.zeros(shape=())
        
        # Add content loss
        layer_features = generated_features[self.content_layers[0]]
        content_features = self.content_features[self.content_layers[0]]
        loss = loss + self.content_weight * self.content_loss(content_features, layer_features)
        
        # Add style loss
        for layer_name in self.style_layers:
            style_features = self.style_features[layer_name]
            combination_features = generated_features[layer_name]
            style_loss = self.style_loss(style_features, combination_features)
            loss += (self.style_weight / len(self.style_layers)) * style_loss
        
        # Add total variation loss
        loss += self.total_variation_weight * self.total_variation_loss(self.generated_image)
        
        return loss
    
    @tf.function
    def train_step(self, optimizer):
        """Perform a single training step."""
        with tf.GradientTape() as tape:
            generated_outputs = self.model(self.generated_image)
            loss = self.compute_loss(generated_outputs)
        
        # Compute gradients and update the generated image
        grads = tape.gradient(loss, [self.generated_image])
        optimizer.apply_gradients(zip(grads, [self.generated_image]))
        
        # Clip the pixel values to the valid range
        self.generated_image.assign(tf.clip_by_value(self.generated_image, 0.0, 255.0))
        
        return loss
    
    def generate(self, epochs=10, steps_per_epoch=100, learning_rate=8.0):
        """Generate the stylized image."""
        # Get the style and content feature representations
        self.style_features = self.model(self.style_image)
        self.content_features = self.model(self.content_image)
        
        # Create optimizer
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        
        # Create output directory if it doesn't exist
        os.makedirs('output', exist_ok=True)
        
        # Training loop
        start_time = time.time()
        for epoch in range(epochs):
            for step in range(steps_per_epoch):
                loss = self.train_step(optimizer)
                if step % 50 == 0:
                    print(f'Epoch {epoch + 1}/{epochs}, Step {step}/{steps_per_epoch}, Loss: {loss:.4f}')
            
            # Save the generated image at the end of each epoch
            img = self.deprocess_image(self.generated_image.numpy())
            output_path = f'output/{self.result_prefix}_at_epoch_{epoch + 1}.png'
            save_img(output_path, img)
            print(f'Saved {output_path}')
        
        print(f'Total time: {time.time() - start_time:.2f}s')
        return self.deprocess_image(self.generated_image.numpy())

def save_input_images(content_path, style_path):
    """Helper function to save content and style images."""
    # Create output directory if it doesn't exist
    os.makedirs('output', exist_ok=True)
    
    # Save content image
    content_img = load_img(content_path, target_size=(400, 600))
    content_img.save('output/content_image.jpg')
    print(f"Saved content image to: output/content_image.jpg")
    
    # Save style image
    style_img = load_img(style_path, target_size=(400, 600))
    style_img.save('output/style_image.jpg')
    print(f"Saved style image to: output/style_image.jpg")

def get_image_path(prompt, default_filename):
    """Helper function to get image path with error handling."""
    while True:
        path = input(f"{prompt} (or press Enter to use '{default_filename}'): ").strip()
        if not path:
            path = default_filename
        if os.path.exists(path):
            return path
        print(f"Error: File '{path}' not found. Please try again.")

def main():
    print("=== Neural Style Transfer ===")
    print("Please provide the paths to your content and style images.")
    print("Place your images in the same directory as this script or provide full paths.\n")
    
    # Get image paths
    content_path = get_image_path("Enter path to content image", "content.jpg")
    style_path = get_image_path("Enter path to style image", "style.jpg")
    
    # Create output directory
    os.makedirs('output', exist_ok=True)
    
    # Save input images
    try:
        save_input_images(content_path, style_path)
    except Exception as e:
        print(f"Error processing images: {e}")
        print("Please make sure the image files are valid and accessible.")
        return
    
    # Create and run the style transfer
    nst = NeuralStyleTransfer(content_path, style_path)
    print("Starting style transfer...")
    result = nst.generate(epochs=5, steps_per_epoch=100, learning_rate=8.0)
    
    # Save the final result
    output_path = 'output/stylized_result.jpg'
    plt.imsave(output_path, result)
    print(f"\nStyle transfer complete!")
    print(f"Final result saved to: {output_path}")
    print("\nYou can find all output files in the 'output' directory.")

if __name__ == "__main__":
    main()