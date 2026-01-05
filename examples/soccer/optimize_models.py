"""
TensorRT Model Optimization Script for Jetson Orin Nano

This script converts YOLO .pt models to TensorRT .engine format for optimized
inference on NVIDIA Jetson devices.
"""

import argparse
import os
import sys
from pathlib import Path
from ultralytics import YOLO


def convert_model_to_tensorrt(
    model_path: str,
    precision: str = 'fp16',
    workspace: int = 4,
    batch_size: int = 1,
    imgsz: int = 640,
    validate: bool = True
) -> str:
    """
    Convert a YOLO model to TensorRT engine format.
    
    Args:
        model_path: Path to the .pt model file
        precision: Precision mode ('fp32', 'fp16', or 'int8')
        workspace: Maximum workspace size in GB for TensorRT
        batch_size: Batch size for the engine
        imgsz: Input image size
        validate: Whether to validate the converted model
        
    Returns:
        Path to the generated .engine file
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    if not model_path.endswith('.pt'):
        raise ValueError("Model file must have .pt extension")
    
    print(f"\n{'='*60}")
    print(f"Converting model: {model_path}")
    print(f"Precision: {precision.upper()}")
    print(f"Image size: {imgsz}")
    print(f"Batch size: {batch_size}")
    print(f"Workspace: {workspace}GB")
    print(f"{'='*60}\n")
    
    # Load the model
    print("Loading model...")
    model = YOLO(model_path)
    
    # Determine output path
    model_dir = os.path.dirname(model_path)
    model_name = os.path.basename(model_path).replace('.pt', '')
    engine_path = os.path.join(model_dir, f"{model_name}-{precision}.engine")
    
    # Export to TensorRT
    print(f"Exporting to TensorRT ({precision.upper()})...")
    try:
        # Set precision-specific parameters
        half = (precision == 'fp16')
        int8 = (precision == 'int8')
        
        export_path = model.export(
            format='engine',
            half=half,
            int8=int8,
            workspace=workspace,
            batch=batch_size,
            imgsz=imgsz,
            verbose=True
        )
        
        print(f"\n✓ Model exported successfully!")
        print(f"  Engine file: {export_path}")
        
        # Validate the model
        if validate:
            print("\nValidating converted model...")
            try:
                # Load the engine and run a test inference
                engine_model = YOLO(export_path)
                print("✓ Engine loaded successfully")
                
                # Create a dummy input for validation
                import numpy as np
                dummy_input = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)
                result = engine_model(dummy_input, verbose=False)
                print("✓ Test inference successful")
                
            except Exception as e:
                print(f"✗ Validation failed: {e}")
                return export_path
        
        return export_path
        
    except Exception as e:
        print(f"\n✗ Export failed: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(
        description='Convert YOLO models to TensorRT for Jetson Orin Nano',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert player detection model to FP16
  python optimize_models.py --model data/football-player-detection.pt --precision fp16
  
  # Convert all models in data directory
  python optimize_models.py --model data/football-player-detection.pt data/football-pitch-detection.pt data/football-ball-detection.pt
  
  # Convert with INT8 precision (requires calibration data)
  python optimize_models.py --model data/football-player-detection.pt --precision int8
  
  # Convert with custom image size
  python optimize_models.py --model data/football-player-detection.pt --imgsz 1280

Notes:
  - FP16 is recommended for Jetson Orin Nano (good balance of speed and accuracy)
  - INT8 provides maximum speed but may reduce accuracy
  - Ensure you have enough disk space (engines can be large)
  - First conversion may take several minutes
        """
    )
    
    parser.add_argument(
        '--model',
        type=str,
        nargs='+',
        required=True,
        help='Path(s) to .pt model file(s) to convert'
    )
    parser.add_argument(
        '--precision',
        type=str,
        choices=['fp32', 'fp16', 'int8'],
        default='fp16',
        help='Precision mode (default: fp16)'
    )
    parser.add_argument(
        '--workspace',
        type=int,
        default=4,
        help='Maximum workspace size in GB (default: 4)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=1,
        help='Batch size for inference (default: 1)'
    )
    parser.add_argument(
        '--imgsz',
        type=int,
        default=640,
        help='Input image size (default: 640)'
    )
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='Skip validation of converted models'
    )
    
    args = parser.parse_args()
    
    # Convert each model
    success_count = 0
    fail_count = 0
    
    for model_path in args.model:
        try:
            engine_path = convert_model_to_tensorrt(
                model_path=model_path,
                precision=args.precision,
                workspace=args.workspace,
                batch_size=args.batch_size,
                imgsz=args.imgsz,
                validate=not args.no_validate
            )
            success_count += 1
            print(f"\n✓ Successfully converted: {model_path}")
            print(f"  → {engine_path}\n")
            
        except Exception as e:
            fail_count += 1
            print(f"\n✗ Failed to convert: {model_path}")
            print(f"  Error: {e}\n")
    
    # Summary
    print(f"\n{'='*60}")
    print(f"Conversion Summary:")
    print(f"  Success: {success_count}")
    print(f"  Failed: {fail_count}")
    print(f"{'='*60}\n")
    
    if success_count > 0:
        print("Usage:")
        print("  Use the .engine files with main.py by specifying the path:")
        print("  python main.py --source_video_path video.mp4 --device cuda --mode PLAYER_TRACKING")
        print("  The script will automatically detect and use .engine files if available.\n")
    
    return 0 if fail_count == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
