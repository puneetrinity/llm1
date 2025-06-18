#!/usr/bin/env python3
"""
quick_websocket_fix.py - Quick WebSocket Authentication Fix
This will disable authentication to resolve your 401 errors immediately
"""

import os
import sys
from pathlib import Path


def update_env_file():
    """Update .env file to disable authentication"""
    env_file = Path(".env")

    if not env_file.exists():
        print("❌ .env file not found!")
        return False

    try:
        # Read current content
        with open(env_file, 'r') as f:
            lines = f.readlines()

        # Update the lines
        updated_lines = []
        changes_made = []

        for line in lines:
            original_line = line

            # Disable authentication
            if line.strip().startswith("ENABLE_AUTH=true"):
                line = line.replace("ENABLE_AUTH=true", "ENABLE_AUTH=false")
                changes_made.append("✅ Disabled authentication")

            # Enable WebSocket dashboard for better experience
            elif line.strip().startswith("ENABLE_WEBSOCKET_DASHBOARD=false"):
                line = line.replace(
                    "ENABLE_WEBSOCKET_DASHBOARD=false", "ENABLE_WEBSOCKET_DASHBOARD=true")
                changes_made.append("✅ Enabled WebSocket dashboard")

            # Enable WebSocket
            elif line.strip().startswith("ENABLE_WEBSOCKET=false"):
                line = line.replace("ENABLE_WEBSOCKET=false",
                                    "ENABLE_WEBSOCKET=true")
                changes_made.append("✅ Enabled WebSocket")

            updated_lines.append(line)

        # Write back to file
        with open(env_file, 'w') as f:
            f.writelines(updated_lines)

        # Show what was changed
        if changes_made:
            print("🔧 Configuration updated:")
            for change in changes_made:
                print(f"   {change}")
        else:
            print("ℹ️ No changes needed - configuration already correct")

        return True

    except Exception as e:
        print(f"❌ Error updating .env file: {e}")
        return False


def show_current_config():
    """Show current configuration"""
    env_file = Path(".env")

    if not env_file.exists():
        print("❌ .env file not found!")
        return

    print("\n📄 Current .env configuration:")
    try:
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if any(key in line for key in ["ENABLE_AUTH", "ENABLE_WEBSOCKET", "DEFAULT_API_KEY"]):
                    if line and not line.startswith("#"):
                        print(f"   {line}")
    except Exception as e:
        print(f"❌ Error reading .env file: {e}")


def main():
    """Main function"""
    print("🔐 Quick WebSocket Authentication Fix")
    print("=" * 40)
    print("This will disable authentication to resolve 401 errors")
    print()

    # Show current config
    show_current_config()

    # Ask for confirmation
    print()
    response = input(
        "🤔 Disable authentication to fix WebSocket errors? (y/n): ").lower().strip()

    if response in ['y', 'yes']:
        print("\n🔄 Updating configuration...")

        if update_env_file():
            print("\n✅ Configuration updated successfully!")
            print()
            print("🚀 Next steps:")
            print("1. Restart your server:")
            print("   python main_master.py")
            print()
            print("2. Test your dashboard:")
            print("   http://localhost:8001/app/")
            print()
            print("3. Your 401 WebSocket errors should be resolved!")
            print()
            print("⚠️ Note: Authentication is now disabled.")
            print("   This is fine for development, but enable it for production.")

            # Show final config
            show_current_config()

        else:
            print("❌ Failed to update configuration")
            return False

    else:
        print("❌ Operation cancelled")
        print("\nAlternative: Set a proper API key in .env:")
        print("   DEFAULT_API_KEY=sk-your-secure-key-here")
        return False

    return True


if __name__ == "__main__":
    try:
        success = main()
        if not success:
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)
