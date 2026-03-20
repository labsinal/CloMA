"""
Installation and configuration script for CloMA napari plugin
"""

import subprocess
import sys
from pathlib import Path


def install_plugin():
    """Install the CloMA napari plugin"""
    plugin_dir = Path(__file__).parent

    print("="*60)
    print("Installing CloMA Napari Plugin")
    print("="*60)

    # Install in editable mode
    print("\nStep 1: Installing plugin in development mode...")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-e", str(plugin_dir)],
        capture_output=True,
        text=True
    )

    if result.returncode == 0:
        print("✓ Plugin installed successfully!")
    else:
        print("✗ Installation failed!")
        print(result.stderr)
        return False

    # Verify napari can find it
    print("\nStep 2: Verifying napari can find the plugin...")
    try:
        import napari.plugins
        plugins = napari.plugins.menu
        if "CloMA" in [p[0] for p in plugins]:
            print("✓ Plugin found by napari!")
        else:
            print("⚠ Plugin not yet visible in napari (may need restart)")
    except Exception as e:
        print(f"⚠ Could not verify: {e}")

    print("\n" + "="*60)
    print("Installation Complete!")
    print("="*60)
    print("\nNext steps:")
    print("1. Start napari: napari")
    print("2. Open an image: File > Open")
    print("3. Find 'CloMA Tools' in the dock widgets")
    print("\nFor more help, see QUICKSTART.md or README.md")
    print("="*60)

    return True


if __name__ == "__main__":
    install_plugin()
