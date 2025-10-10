import onnxruntime as rt
import numpy as np
import struct 
import time

def load_mnist_image(filename):
    """Loads a single MNIST image from a .ubyte file."""
    with open(filename, 'rb') as f:
        # Read image data as bytes
        image_data = f.read(28 * 28) 
        # Convert to numpy array, reshape, and normalize
        image = np.frombuffer(image_data, dtype=np.uint8).reshape(1, 1, 28, 28).astype(np.float32)
    return image

def load_mnist_dataset(images_file, labels_file=None):
    """
    Load the full MNIST test dataset from IDX format files.
    
    Args:
        images_file: Path to t10k-images.idx3-ubyte
        labels_file: Path to t10k-labels.idx1-ubyte (optional)
    
    Returns:
        images: numpy array of shape (num_images, 1, 28, 28)
        labels: numpy array of shape (num_images,) if labels_file provided, else None
    """
    
    # Load images
    with open(images_file, 'rb') as f:
        # Read IDX header for images
        magic, num_images, rows, cols = struct.unpack('>IIII', f.read(16))
        
        if magic != 2051:
            raise ValueError(f"Invalid magic number in images file: {magic}")
        
        print(f"Loading {num_images} images of size {rows}x{cols}")
        
        # Read all image data
        image_data = f.read()
        images = np.frombuffer(image_data, dtype=np.uint8)
        images = images.reshape(num_images, 1, rows, cols).astype(np.float32)
        
        # Normalize to [0, 1] range (MNIST images are 0-255)
        images = images / 255.0
    
    # Load labels if provided
    labels = None
    if labels_file:
        with open(labels_file, 'rb') as f:
            # Read IDX header for labels
            magic, num_labels = struct.unpack('>II', f.read(8))
            
            if magic != 2049:
                raise ValueError(f"Invalid magic number in labels file: {magic}")
            
            if num_labels != num_images:
                raise ValueError(f"Number of labels ({num_labels}) doesn't match number of images ({num_images})")
            
            # Read all label data
            label_data = f.read()
            labels = np.frombuffer(label_data, dtype=np.uint8)
    
    return images, labels

def run_batch_inference(sess, images, batch_size=100):
    """
    Run inference on images one by one (since model expects batch_size=1).
    
    Args:
        sess: ONNX Runtime session
        images: numpy array of images
        batch_size: number of images to process before printing progress (not actual batch size)
    
    Returns:
        predictions: list of predicted classes
        probabilities: list of prediction probabilities
    """
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    
    num_images = images.shape[0]
    predictions = []
    probabilities = []
    
    # Process images one by one since model expects batch_size=1
    for i in range(num_images):
        # Get single image and ensure it has shape (1, 1, 28, 28)
        single_image = images[i:i+1]  # Keep batch dimension
        
        # Run inference on single image
        result = sess.run([output_name], {input_name: single_image})
        probs = result[0][0]  # Get probabilities for this single image
        
        # Get prediction
        prediction = np.argmax(probs)
        
        predictions.append(prediction)
        probabilities.append(probs)
        
        # Print progress every batch_size images
        if (i + 1) % batch_size == 0 or (i + 1) == num_images:
            print(f"Processed {i + 1}/{num_images} images...")
    
    return predictions, probabilities

def evaluate_accuracy(predictions, true_labels):
    """Calculate accuracy metrics."""
    correct = np.sum(predictions == true_labels)
    total = len(true_labels)
    accuracy = correct / total
    
    # Per-class accuracy
    class_accuracy = {}
    for digit in range(10):
        mask = true_labels == digit
        if np.sum(mask) > 0:
            class_correct = np.sum((predictions == true_labels) & mask)
            class_total = np.sum(mask)
            class_accuracy[digit] = class_correct / class_total
    
    return accuracy, class_accuracy

# Load the ONNX model
print("Loading ONNX model...")
sess = rt.InferenceSession("../models/mnist_ffn.onnx")

# Get input and output names
input_name = sess.get_inputs()[0].name
output_name = sess.get_outputs()[0].name
print("Input name:", input_name)
print("Output name:", output_name)

def main():
    """Main function to run inference on full MNIST test dataset."""
    
    # Option 1: Test individual images (your current approach)
    print("\n=== Testing individual images ===")
    for i in range(min(10, 5)):  # Test first 5 only to save time
        try:
            image = load_mnist_image(f"../inputs/digit_{i}_test.ubyte")
            result = sess.run([output_name], {input_name: image})
            predicted_class = np.argmax(result)
            confidence = np.max(result)
            print(f"digit_{i}_test.ubyte: predicted={predicted_class}, confidence={confidence:.4f}")
        except FileNotFoundError:
            print(f"digit_{i}_test.ubyte not found, skipping...")
    
    # Option 2: Full MNIST test dataset
    print("\n=== Testing full MNIST dataset ===")
    
    try:
        # Load the full MNIST test dataset
        print("Loading MNIST test dataset...")
        start_time = time.time()
        
        images, labels = load_mnist_dataset(
            images_file="../inputs/t10k-images-idx3-ubyte",
            labels_file="../inputs/t10k-labels-idx1-ubyte"  # Optional, comment out if you don't have labels
        )
        
        load_time = time.time() - start_time
        print(f"Loaded {len(images)} images in {load_time:.2f} seconds")
        
        # Run inference on all images
        print("Running inference...")
        start_time = time.time()
        
        predictions, probabilities = run_batch_inference(sess, images, batch_size=1000)
        
        inference_time = time.time() - start_time
        print(f"Inference completed in {inference_time:.2f} seconds")
        print(f"Average time per image: {(inference_time / len(images) * 1000):.2f} ms")
        
        # Show some example predictions
        print("\n=== Sample predictions ===")
        for i in range(min(10, len(predictions))):
            pred = predictions[i]
            prob = probabilities[i]
            confidence = np.max(prob)
            
            if labels is not None:
                true_label = labels[i]
                correct = "✓" if pred == true_label else "✗"
                print(f"Image {i}: true={true_label}, pred={pred}, confidence={confidence:.4f} {correct}")
            else:
                print(f"Image {i}: pred={pred}, confidence={confidence:.4f}")
        
        # Calculate accuracy if we have labels
        if labels is not None:
            print("\n=== Accuracy Results ===")
            accuracy, class_accuracy = evaluate_accuracy(np.array(predictions), labels)
            print(f"Overall accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
            print(f"Correct predictions: {np.sum(np.array(predictions) == labels)}/{len(labels)}")
            
            print("\nPer-class accuracy:")
            for digit, acc in class_accuracy.items():
                print(f"  Digit {digit}: {acc:.4f} ({acc*100:.2f}%)")
        
        # Save results (optional)
        print("\n=== Saving results ===")
        results_file = "../results/inference_results.txt"
        try:
            import os
            os.makedirs("../results", exist_ok=True)
            
            with open(results_file, 'w') as f:
                f.write(f"MNIST Inference Results\n")
                f.write(f"======================\n")
                f.write(f"Total images: {len(images)}\n")
                f.write(f"Inference time: {inference_time:.2f} seconds\n")
                f.write(f"Time per image: {(inference_time / len(images) * 1000):.2f} ms\n")
                
                if labels is not None:
                    f.write(f"Overall accuracy: {accuracy:.4f}\n\n")
                    f.write("Per-class accuracy:\n")
                    for digit, acc in class_accuracy.items():
                        f.write(f"Digit {digit}: {acc:.4f}\n")
                
                f.write(f"\nFirst 100 predictions:\n")
                for i in range(min(100, len(predictions))):
                    if labels is not None:
                        f.write(f"{i}: true={labels[i]}, pred={predictions[i]}\n")
                    else:
                        f.write(f"{i}: pred={predictions[i]}\n")
            
            print(f"Results saved to {results_file}")
        except Exception as e:
            print(f"Could not save results: {e}")
    
    except FileNotFoundError as e:
        print(f"MNIST dataset file not found: {e}")
        print("Make sure t10k-images.idx3-ubyte is in the ../inputs/ directory")
    except Exception as e:
        print(f"Error processing MNIST dataset: {e}")

if __name__ == "__main__":
    main()