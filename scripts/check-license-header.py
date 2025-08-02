#!/usr/bin/env python3
"""
License header checker for Agent Mesh Federated Runtime.
Ensures all Python files have proper license headers.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional


# Expected license headers
MIT_HEADER = '''"""
Agent Mesh Federated Runtime
Licensed under the MIT License. See LICENSE file for details.

Copyright (c) 2024 Terragon Labs
"""'''

APACHE_HEADER = '''"""
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Copyright (c) 2024 Terragon Labs
"""'''

# Default license type based on LICENSE file
DEFAULT_LICENSE = "MIT"

# Files to exclude from license header checks
EXCLUDE_PATTERNS = [
    "*_pb2.py",
    "*_pb2_grpc.py",
    "__init__.py",
    "conftest.py",
    "setup.py",
    "manage.py",
]

# Directories to exclude
EXCLUDE_DIRS = [
    "tests",
    "migrations", 
    "venv",
    ".venv",
    "node_modules",
    "build",
    "dist",
    "__pycache__",
]


def get_license_header(license_type: str) -> str:
    """Get the appropriate license header based on type."""
    if license_type.upper() == "APACHE":
        return APACHE_HEADER
    else:
        return MIT_HEADER


def should_exclude_file(file_path: Path) -> bool:
    """Check if file should be excluded from license header check."""
    # Check if file matches exclude patterns
    for pattern in EXCLUDE_PATTERNS:
        if file_path.match(pattern):
            return True
    
    # Check if file is in excluded directory
    for exclude_dir in EXCLUDE_DIRS:
        if exclude_dir in file_path.parts:
            return True
    
    return False


def has_license_header(file_path: Path, expected_header: str) -> bool:
    """Check if file has the expected license header."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Remove shebang line if present
        lines = content.split('\n')
        if lines and lines[0].startswith('#!'):
            content = '\n'.join(lines[1:])
        
        # Normalize whitespace for comparison
        content_normalized = content.strip()
        header_normalized = expected_header.strip()
        
        return content_normalized.startswith(header_normalized)
    
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return False


def add_license_header(file_path: Path, license_header: str) -> bool:
    """Add license header to file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Preserve shebang line if present
        lines = content.split('\n')
        shebang = ""
        if lines and lines[0].startswith('#!'):
            shebang = lines[0] + '\n'
            content = '\n'.join(lines[1:])
        
        # Add header at the beginning
        new_content = shebang + license_header + '\n\n' + content.lstrip()
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        return True
    
    except Exception as e:
        print(f"Error writing to {file_path}: {e}")
        return False


def find_python_files(root_dir: Path) -> List[Path]:
    """Find all Python files that should have license headers."""
    python_files = []
    
    for file_path in root_dir.rglob("*.py"):
        if not should_exclude_file(file_path):
            python_files.append(file_path)
    
    return python_files


def detect_license_type(root_dir: Path) -> str:
    """Detect license type from LICENSE file."""
    license_file = root_dir / "LICENSE"
    if not license_file.exists():
        return DEFAULT_LICENSE
    
    try:
        with open(license_file, 'r', encoding='utf-8') as f:
            content = f.read().upper()
        
        if "APACHE" in content:
            return "APACHE"
        elif "MIT" in content:
            return "MIT"
        else:
            return DEFAULT_LICENSE
    
    except Exception:
        return DEFAULT_LICENSE


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Check and add license headers to Python files"
    )
    parser.add_argument(
        "--root", 
        type=Path, 
        default=Path.cwd(),
        help="Root directory to scan (default: current directory)"
    )
    parser.add_argument(
        "--license-type",
        choices=["MIT", "APACHE"],
        help="License type to use (default: auto-detect from LICENSE file)"
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Add missing license headers"
    )
    parser.add_argument(
        "--check-only",
        action="store_true", 
        help="Only check, don't fix (exit 1 if headers missing)"
    )
    
    args = parser.parse_args()
    
    # Determine license type
    license_type = args.license_type or detect_license_type(args.root)
    license_header = get_license_header(license_type)
    
    print(f"Checking license headers ({license_type}) in: {args.root}")
    
    # Find Python files
    python_files = find_python_files(args.root)
    print(f"Found {len(python_files)} Python files to check")
    
    # Check files
    missing_headers = []
    for file_path in python_files:
        if not has_license_header(file_path, license_header):
            missing_headers.append(file_path)
    
    if not missing_headers:
        print("✅ All files have proper license headers")
        return 0
    
    print(f"❌ {len(missing_headers)} files missing license headers:")
    for file_path in missing_headers:
        print(f"  - {file_path}")
    
    if args.check_only:
        return 1
    
    if args.fix:
        print(f"\n🔧 Adding license headers...")
        fixed_count = 0
        for file_path in missing_headers:
            if add_license_header(file_path, license_header):
                print(f"  ✅ Added header to {file_path}")
                fixed_count += 1
            else:
                print(f"  ❌ Failed to add header to {file_path}")
        
        print(f"\n📊 Summary: {fixed_count}/{len(missing_headers)} files fixed")
        return 0 if fixed_count == len(missing_headers) else 1
    
    else:
        print("\n💡 Run with --fix to add missing headers")
        return 1


if __name__ == "__main__":
    sys.exit(main())