"""
Master script to generate all figures for Model Selection and Evaluation.

This script executes all figure generation scripts in the correct order.
"""

import sys
import subprocess
import os

def run_script(script_name):
    """Run a Python script and report status."""
    print(f"\n{'='*60}")
    print(f"Running {script_name}...")
    print('='*60)

    try:
        result = subprocess.run(
            [sys.executable, script_name],
            capture_output=True,
            text=True,
            check=True
        )
        print(result.stdout)
        if result.stderr:
            print("Warnings/Errors:", result.stderr)
        print(f"✓ {script_name} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error running {script_name}")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False


def main():
    """Execute all figure generation scripts."""
    print("="*60)
    print("MODEL SELECTION AND EVALUATION - Figure Generation")
    print("="*60)

    # Get script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    # List of scripts to run
    scripts = [
        'core_methods.py',
        'regularization_methods.py',
        'validation_methods.py'
    ]

    # Track results
    results = {}

    # Run each script
    for script in scripts:
        results[script] = run_script(script)

    # Final summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    for script, success in results.items():
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"{script:30s} {status}")

    # Overall status
    all_success = all(results.values())
    print("\n" + "="*60)
    if all_success:
        print("✓ ALL FIGURES GENERATED SUCCESSFULLY!")
    else:
        print("✗ SOME SCRIPTS FAILED - Please check errors above")
    print("="*60)

    return 0 if all_success else 1


if __name__ == "__main__":
    sys.exit(main())
