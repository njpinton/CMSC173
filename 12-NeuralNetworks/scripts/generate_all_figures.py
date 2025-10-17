#!/usr/bin/env python3
"""
Generate All Neural Networks Figures
====================================

Master script to generate all visualizations for the Neural Networks
educational package. This script runs all individual scripts and creates
the complete set of figures needed for the course materials.

Author: CMSC 173 Machine Learning Course
"""

import os
import sys
import importlib.util
from pathlib import Path
import time

def import_script(script_path):
    """Import a Python script as a module"""
    spec = importlib.util.spec_from_file_location("module", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def run_script(script_name, script_path):
    """Run a script and handle any errors"""
    print(f"\n{'='*60}")
    print(f"Running {script_name}...")
    print(f"{'='*60}")

    start_time = time.time()

    try:
        # Import and run the script
        module = import_script(script_path)
        if hasattr(module, 'main'):
            module.main()
        else:
            print(f"Warning: {script_name} does not have a main() function")

        elapsed_time = time.time() - start_time
        print(f"✅ {script_name} completed successfully in {elapsed_time:.2f} seconds")
        return True

    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"❌ Error in {script_name} after {elapsed_time:.2f} seconds:")
        print(f"   {str(e)}")
        return False

def create_figures_directory():
    """Ensure the figures directory exists"""
    figures_dir = Path("../figures")
    figures_dir.mkdir(exist_ok=True)
    print(f"📁 Figures directory: {figures_dir.absolute()}")

def main():
    """Main function to run all figure generation scripts"""
    print("🎯 Neural Networks Figure Generation")
    print("=" * 60)

    # Create figures directory
    create_figures_directory()

    # Define scripts to run
    scripts = [
        ("Neural Network Basics", "neural_network_basics.py"),
        ("Forward & Backpropagation", "forward_backprop_demo.py"),
        ("Regularization Techniques", "regularization_techniques.py"),
    ]

    # Track results
    results = {}
    total_start_time = time.time()

    # Run each script
    for script_name, script_file in scripts:
        script_path = Path(script_file)

        if not script_path.exists():
            print(f"❌ Script not found: {script_path}")
            results[script_name] = False
            continue

        success = run_script(script_name, script_path)
        results[script_name] = success

    # Summary
    total_elapsed = time.time() - total_start_time
    print(f"\n{'='*60}")
    print("📊 GENERATION SUMMARY")
    print(f"{'='*60}")

    successful = sum(results.values())
    total = len(results)

    for script_name, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"{script_name:30} {status}")

    print(f"\nTotal time: {total_elapsed:.2f} seconds")
    print(f"Success rate: {successful}/{total} ({100*successful/total:.1f}%)")

    if successful == total:
        print("\n🎉 All figures generated successfully!")
        print("\nGenerated figures:")

        # List all generated figures
        figures_dir = Path("../figures")
        if figures_dir.exists():
            png_files = sorted(figures_dir.glob("*.png"))
            for i, fig_file in enumerate(png_files, 1):
                print(f"  {i:2d}. {fig_file.name}")

            print(f"\nTotal figures: {len(png_files)}")

        print("\n📋 Next steps:")
        print("1. Review generated figures in ../figures/")
        print("2. Compile LaTeX slides with 'pdflatex neural_networks_slides.tex'")
        print("3. Run Jupyter notebook for interactive content")
        print("4. Check README.md for detailed instructions")

    else:
        failed_scripts = [name for name, success in results.items() if not success]
        print(f"\n⚠️  Some scripts failed: {', '.join(failed_scripts)}")
        print("Please check error messages above and fix issues before proceeding.")
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())