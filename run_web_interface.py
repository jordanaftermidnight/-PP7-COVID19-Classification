#!/usr/bin/env python3
"""
Launcher script for COVID-19 Web Interface
Choose between Streamlit and Flask interfaces
"""

import subprocess
import sys
import os

def run_streamlit():
    """Launch Streamlit interface"""
    print("🚀 Launching Streamlit Web Interface...")
    print("📱 The interface will open in your browser automatically")
    print("🔗 URL: http://localhost:8501")
    print("\n⏹️  Press Ctrl+C to stop the server")
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "web_interface.py",
            "--server.port", "8501",
            "--browser.gatherUsageStats", "false"
        ])
    except KeyboardInterrupt:
        print("\n✅ Streamlit server stopped")

def run_flask():
    """Launch Flask interface"""
    print("🚀 Launching Flask Web Interface...")
    print("📱 Open http://localhost:5000 in your browser")
    print("\n⏹️  Press Ctrl+C to stop the server")
    
    try:
        subprocess.run([sys.executable, "flask_app.py"])
    except KeyboardInterrupt:
        print("\n✅ Flask server stopped")

def main():
    print("COVID-19 Chest X-Ray Classifier - Web Interface Launcher")
    print("=" * 60)
    
    # Check if models exist
    model_paths = [
        'models/covid_classifier_extended.pth',
        'models/covid_classifier.pth'
    ]
    
    model_found = any(os.path.exists(path) for path in model_paths)
    
    if not model_found:
        print("❌ No trained model found!")
        print("Please run one of the training scripts first:")
        print("  - python3 train_model.py")
        print("  - python3 extended_training.py")
        return
    
    print("✅ Trained model found")
    print("\nChoose your preferred web interface:")
    print("\n1. 🎨 Streamlit Interface (Recommended)")
    print("   - Beautiful, interactive interface")
    print("   - Real-time Grad-CAM visualization") 
    print("   - Progress bars and metrics")
    print("   - Medical-grade UI design")
    
    print("\n2. ⚡ Flask Interface")
    print("   - Fast, lightweight interface")
    print("   - Simple upload and predict")
    print("   - Works on any device")
    print("   - No additional dependencies")
    
    print("\n3. 🔧 Demo Mode (Use sample images)")
    print("   - Test with existing dataset images")
    print("   - No file upload needed")
    
    while True:
        try:
            choice = input("\nEnter your choice (1, 2, or 3): ").strip()
            
            if choice == '1':
                run_streamlit()
                break
            elif choice == '2':
                run_flask()
                break
            elif choice == '3':
                demo_mode()
                break
            else:
                print("Invalid choice. Please enter 1, 2, or 3.")
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break

def demo_mode():
    """Run demo with sample images"""
    print("\n🎬 Demo Mode - Testing with sample images")
    
    # Check if we have sample images
    covid_dir = 'data/COVID'
    normal_dir = 'data/Normal'
    
    if not (os.path.exists(covid_dir) and os.path.exists(normal_dir)):
        print("❌ No sample images found in data/ directory")
        print("Please run the dataset download first or use the web interface with your own images")
        return
    
    print("📸 Found sample images in data/ directory")
    print("🔄 Running Grad-CAM visualization on sample images...")
    
    try:
        subprocess.run([sys.executable, "grad_cam_visualization.py"])
        print("\n✅ Demo completed! Check the generated visualization files.")
    except Exception as e:
        print(f"❌ Error running demo: {e}")

if __name__ == "__main__":
    main()