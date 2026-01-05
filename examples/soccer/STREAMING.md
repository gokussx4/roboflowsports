# RTMP Streaming Guide for Jetson Orin Nano

This guide explains how to set up and use the Soccer AI system for real-time RTMP stream processing on NVIDIA Jetson Orin Nano devices.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Model Optimization](#model-optimization)
- [nginx RTMP Setup](#nginx-rtmp-setup)
- [Running the Stream Processor](#running-the-stream-processor)
- [Performance Tuning](#performance-tuning)
- [Troubleshooting](#troubleshooting)
- [Example Commands](#example-commands)

## Prerequisites

### Hardware Requirements

- **NVIDIA Jetson Orin Nano** (8GB recommended)
- Stable power supply (15W or 25W mode)
- Active cooling (recommended for sustained performance)
- Minimum 64GB storage (for models and recordings)
- Network connection for RTMP streaming

### Software Requirements

- **JetPack 5.0+** (includes CUDA, cuDNN, TensorRT)
- **Python 3.8+**
- **OpenCV with CUDA support**
- **FFmpeg with NVENC support**
- **nginx with RTMP module**

## Installation

### 1. Install System Dependencies

```bash
# Update system packages
sudo apt-get update
sudo apt-get upgrade -y

# Install FFmpeg with NVIDIA hardware encoding support
sudo apt-get install -y ffmpeg

# Verify FFmpeg has nvenc support
ffmpeg -encoders | grep nvenc

# Install nginx with RTMP module
sudo apt-get install -y libnginx-mod-rtmp nginx
```

### 2. Install Python Packages

```bash
# Navigate to the soccer example directory
cd examples/soccer

# Install base requirements
pip install -r requirements.txt

# Install additional streaming dependencies
pip install opencv-python

# For Jetson, you may need to install opencv from nvidia-pyindex
# pip install --extra-index-url https://pypi.ngc.nvidia.com nvidia-pyindex
# pip install opencv-python==4.5.5.64
```

### 3. Verify GPU Support

```bash
# Check CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Check GPU information
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

### 4. Download Models

```bash
# Run the setup script to download pre-trained models
./setup.sh
```

## Model Optimization

Convert YOLO models to TensorRT for optimal performance on Jetson.

### Basic Conversion (FP16 - Recommended)

```bash
# Convert player detection model
python optimize_models.py \
    --model data/football-player-detection.pt \
    --precision fp16

# Convert pitch detection model
python optimize_models.py \
    --model data/football-pitch-detection.pt \
    --precision fp16

# Convert ball detection model
python optimize_models.py \
    --model data/football-ball-detection.pt \
    --precision fp16
```

### Convert All Models at Once

```bash
python optimize_models.py \
    --model data/football-player-detection.pt \
           data/football-pitch-detection.pt \
           data/football-ball-detection.pt \
    --precision fp16
```

### Advanced Options

```bash
# Convert with INT8 for maximum speed (may reduce accuracy)
python optimize_models.py \
    --model data/football-player-detection.pt \
    --precision int8

# Convert with custom input size
python optimize_models.py \
    --model data/football-player-detection.pt \
    --imgsz 1280 \
    --precision fp16

# Skip validation (faster conversion)
python optimize_models.py \
    --model data/football-player-detection.pt \
    --precision fp16 \
    --no-validate
```

**Note:** The first conversion may take 5-10 minutes. Subsequent conversions are faster.

## nginx RTMP Setup

### 1. Configure nginx

```bash
# Copy the sample configuration
sudo cp nginx-rtmp.conf /etc/nginx/nginx.conf

# Create recording directory
sudo mkdir -p /tmp/recordings
sudo chmod 777 /tmp/recordings

# Test configuration
sudo nginx -t

# Restart nginx
sudo systemctl restart nginx

# Enable nginx to start on boot
sudo systemctl enable nginx
```

### 2. Verify nginx is Running

```bash
# Check nginx status
sudo systemctl status nginx

# Check RTMP port
netstat -an | grep 1935

# View nginx logs
sudo tail -f /var/log/nginx/error.log
```

### 3. Test RTMP Server

```bash
# Push a test stream from a video file
ffmpeg -re -i data/2e57b9_0.mp4 \
    -c:v libx264 -preset ultrafast \
    -f flv rtmp://localhost/live/test

# In another terminal, play the stream
ffplay rtmp://localhost/live/test

# View statistics in browser
# Open http://localhost:8080/stat
```

## Running the Stream Processor

### Basic Streaming Mode

```bash
# Process RTMP stream with player tracking
python main.py \
    --stream_mode \
    --input_rtmp_url rtmp://localhost/live/input \
    --output_rtmp_url rtmp://localhost/processed/output \
    --device cuda \
    --mode PLAYER_TRACKING
```

### Real-time Mode (Single-Pass)

For modes that normally require two passes (TEAM_CLASSIFICATION, RADAR), use `--realtime` flag:

```bash
python main.py \
    --stream_mode \
    --input_rtmp_url rtmp://localhost/live/input \
    --output_rtmp_url rtmp://localhost/processed/output \
    --device cuda \
    --mode TEAM_CLASSIFICATION \
    --realtime
```

### Limit Input Resolution

For better performance, limit the input resolution:

```bash
python main.py \
    --stream_mode \
    --input_rtmp_url rtmp://localhost/live/input \
    --output_rtmp_url rtmp://localhost/processed/output \
    --device cuda \
    --mode PLAYER_TRACKING \
    --max_resolution 1280
```

### Complete Example Pipeline

```bash
# Terminal 1: Start the stream processor
python main.py \
    --stream_mode \
    --input_rtmp_url rtmp://localhost/live/camera1 \
    --output_rtmp_url rtmp://localhost/processed/output \
    --device cuda \
    --mode PLAYER_TRACKING \
    --realtime \
    --max_resolution 1280

# Terminal 2: Push input stream from file (for testing)
ffmpeg -re -i data/2e57b9_0.mp4 \
    -c:v libx264 -preset ultrafast -tune zerolatency \
    -f flv rtmp://localhost/live/camera1

# Terminal 3: View the processed output
ffplay rtmp://localhost/processed/output
```

## Performance Tuning

### Power Mode Settings

```bash
# Check current power mode
sudo nvpmodel -q

# Set to maximum performance (25W mode)
sudo nvpmodel -m 0

# Set to balanced mode (15W)
sudo nvpmodel -m 1

# Enable maximum CPU/GPU clocks
sudo jetson_clocks
```

### Resolution Guidelines

| Mode | Recommended Max Resolution | Expected FPS |
|------|---------------------------|--------------|
| PLAYER_DETECTION | 1920x1080 | 25-30 FPS |
| PLAYER_TRACKING | 1920x1080 | 20-25 FPS |
| BALL_DETECTION | 1280x720 | 20-25 FPS |
| TEAM_CLASSIFICATION | 1280x720 | 15-20 FPS |
| RADAR | 1280x720 | 10-15 FPS |

### Memory Management

```bash
# Monitor GPU memory usage
watch -n 1 tegrastats

# Clear cache before starting
sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'

# Set GPU memory fraction in Python (add to main.py if needed)
import torch
torch.cuda.set_per_process_memory_fraction(0.8)
```

### Optimization Checklist

- [ ] Use TensorRT .engine models (2-3x speedup)
- [ ] Enable FP16 precision mode
- [ ] Set Jetson to maximum power mode
- [ ] Enable jetson_clocks
- [ ] Limit input resolution with --max_resolution
- [ ] Use --realtime flag for single-pass processing
- [ ] Enable hardware encoding (h264_nvenc) in stream_processor
- [ ] Reduce FFmpeg output bitrate if network is limited
- [ ] Monitor temperature and ensure adequate cooling

## Troubleshooting

### Common Issues

#### 1. "Failed to open input stream"

**Cause:** nginx not running or incorrect RTMP URL

**Solution:**
```bash
# Check nginx status
sudo systemctl status nginx

# Restart nginx
sudo systemctl restart nginx

# Verify stream is being published
ffprobe rtmp://localhost/live/input
```

#### 2. "h264_nvenc not found"

**Cause:** FFmpeg doesn't have NVENC support

**Solution:**
```bash
# Check FFmpeg encoders
ffmpeg -encoders | grep nvenc

# If not present, reinstall FFmpeg from NVIDIA sources
# or disable hardware encoding in stream_processor.py
```

#### 3. Low FPS / Frame Drops

**Cause:** Processing too slow for input stream

**Solutions:**
- Reduce input resolution: `--max_resolution 1280`
- Use TensorRT engines: Run `optimize_models.py`
- Enable real-time mode: `--realtime`
- Set maximum power mode: `sudo nvpmodel -m 0`
- Use simpler mode: PLAYER_TRACKING instead of RADAR

#### 4. High Latency

**Cause:** Frame buffering or slow processing

**Solutions:**
- Reduce buffer sizes in stream_processor.py (max_queue_size)
- Use lower resolution input
- Optimize FFmpeg encoding parameters
- Check network bandwidth

#### 5. CUDA Out of Memory

**Cause:** Models too large for available GPU memory

**Solutions:**
```bash
# Reduce batch size in model inference
# Use smaller input resolution
# Close other GPU applications
# Monitor with: tegrastats
```

#### 6. Stream Keeps Disconnecting

**Cause:** Network issues or timeout

**Solutions:**
- Check network stability
- Increase reconnect_delay in stream_processor
- Verify nginx configuration
- Check nginx error logs: `sudo tail -f /var/log/nginx/error.log`

### Debug Mode

Enable verbose logging:

```python
# Add to beginning of main.py or stream_processor.py
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Performance Monitoring

```bash
# Monitor GPU usage
tegrastats

# Monitor network usage
iftop

# Monitor process resources
htop

# Check stream statistics
curl http://localhost:8080/stat
```

## Example Commands

### 1. Basic Player Tracking Stream

```bash
python main.py \
    --stream_mode \
    --input_rtmp_url rtmp://localhost/live/game \
    --output_rtmp_url rtmp://localhost/processed/game \
    --device cuda \
    --mode PLAYER_TRACKING
```

### 2. Real-time Team Classification

```bash
python main.py \
    --stream_mode \
    --input_rtmp_url rtmp://localhost/live/game \
    --output_rtmp_url rtmp://localhost/processed/game \
    --device cuda \
    --mode TEAM_CLASSIFICATION \
    --realtime \
    --max_resolution 1280
```

### 3. Ball Detection with Low Latency

```bash
python main.py \
    --stream_mode \
    --input_rtmp_url rtmp://localhost/live/game \
    --output_rtmp_url rtmp://localhost/processed/game \
    --device cuda \
    --mode BALL_DETECTION \
    --max_resolution 1280
```

### 4. Radar View (Performance Mode)

```bash
python main.py \
    --stream_mode \
    --input_rtmp_url rtmp://localhost/live/game \
    --output_rtmp_url rtmp://localhost/processed/game \
    --device cuda \
    --mode RADAR \
    --realtime \
    --max_resolution 960
```

### 5. Testing with File Input

```bash
# Simulate RTMP stream from video file
ffmpeg -re -i data/2e57b9_0.mp4 \
    -c:v libx264 -preset ultrafast -tune zerolatency \
    -b:v 4M -maxrate 4M -bufsize 2M \
    -f flv rtmp://localhost/live/test &

# Process the stream
python main.py \
    --stream_mode \
    --input_rtmp_url rtmp://localhost/live/test \
    --output_rtmp_url rtmp://localhost/processed/test \
    --device cuda \
    --mode PLAYER_TRACKING

# View output
ffplay rtmp://localhost/processed/test
```

### 6. Production Setup with Monitoring

```bash
# Start with automatic restart on failure
while true; do
    python main.py \
        --stream_mode \
        --input_rtmp_url rtmp://localhost/live/camera1 \
        --output_rtmp_url rtmp://localhost/processed/output \
        --device cuda \
        --mode PLAYER_TRACKING \
        --realtime \
        --max_resolution 1280
    
    echo "Process crashed. Restarting in 5 seconds..."
    sleep 5
done
```

## Additional Resources

- [NVIDIA Jetson Documentation](https://developer.nvidia.com/embedded/jetson-orin-nano-developer-kit)
- [TensorRT Documentation](https://docs.nvidia.com/deeplearning/tensorrt/)
- [nginx RTMP Module](https://github.com/arut/nginx-rtmp-module)
- [FFmpeg Documentation](https://ffmpeg.org/documentation.html)
- [Ultralytics YOLOv8](https://docs.ultralytics.com/)

## Support

For issues and questions:
- Open an issue on [GitHub](https://github.com/roboflow/sports/issues)
- Check the main [README.md](README.md) for general setup
- Review Jetson forums for hardware-specific issues
