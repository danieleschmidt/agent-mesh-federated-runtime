#!/usr/bin/env python3
"""
Model Size Monitoring Script for Agent Mesh Federated Runtime

Monitors and validates ML model file sizes to prevent accidental commits
of large models that should be stored in Git LFS or external storage.
Used by pre-commit hooks to enforce model size policies.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class ModelSizeChecker:
    """Checks ML model file sizes and enforces size policies."""
    
    # File extensions for ML models
    MODEL_EXTENSIONS = {
        '.pkl', '.pickle',     # Scikit-learn, general Python objects
        '.pth', '.pt',         # PyTorch models
        '.h5', '.hdf5',        # Keras/TensorFlow models
        '.onnx',               # ONNX models
        '.pb',                 # TensorFlow protobuf
        '.tflite',             # TensorFlow Lite
        '.safetensors',        # Hugging Face safe tensors
        '.bin',                # Generic binary model files
        '.joblib',             # Joblib serialized models
        '.model',              # Generic model files
        '.weights',            # Model weights
        '.ckpt',               # Checkpoint files
    }
    
    # Size limits in bytes
    SIZE_LIMITS = {
        'error': 100 * 1024 * 1024,    # 100MB - hard limit
        'warning': 10 * 1024 * 1024,   # 10MB - soft limit
        'info': 1 * 1024 * 1024,       # 1MB - informational
    }
    
    # Allowed directories for larger models
    ALLOWED_LARGE_MODEL_DIRS = {
        'tests/fixtures',
        'tests/data',
        'examples/models',
        'benchmarks/models'
    }
    
    def __init__(self, root_dir: str = "."):
        self.root_dir = Path(root_dir)
        self.violations = []
        self.warnings = []
        self.info_messages = []
    
    def check_files(self, file_paths: List[str]) -> Tuple[bool, Dict]:
        """Check specified files for size violations."""
        results = {
            'violations': [],
            'warnings': [],
            'info': [],
            'checked_files': 0,
            'total_size': 0
        }
        
        for file_path in file_paths:
            path = Path(file_path)
            
            # Skip if file doesn't exist or is not a model file
            if not path.exists() or not self._is_model_file(path):
                continue
            
            results['checked_files'] += 1
            file_size = path.stat().st_size
            results['total_size'] += file_size
            
            # Check size limits
            violation = self._check_file_size(path, file_size)
            if violation:
                if violation['level'] == 'error':
                    results['violations'].append(violation)
                elif violation['level'] == 'warning':
                    results['warnings'].append(violation)
                else:
                    results['info'].append(violation)
        
        return len(results['violations']) == 0, results
    
    def scan_repository(self) -> Tuple[bool, Dict]:
        """Scan entire repository for model files."""
        results = {
            'violations': [],
            'warnings': [],
            'info': [],
            'checked_files': 0,
            'total_size': 0
        }
        
        # Find all model files
        for ext in self.MODEL_EXTENSIONS:
            for path in self.root_dir.rglob(f"*{ext}"):
                # Skip files in .git, __pycache__, etc.
                if any(part.startswith('.') for part in path.parts):
                    continue
                
                results['checked_files'] += 1
                file_size = path.stat().st_size
                results['total_size'] += file_size
                
                violation = self._check_file_size(path, file_size)
                if violation:
                    if violation['level'] == 'error':
                        results['violations'].append(violation)
                    elif violation['level'] == 'warning':
                        results['warnings'].append(violation)
                    else:
                        results['info'].append(violation)
        
        return len(results['violations']) == 0, results
    
    def _is_model_file(self, path: Path) -> bool:
        """Check if file is a model file based on extension."""
        return path.suffix.lower() in self.MODEL_EXTENSIONS
    
    def _check_file_size(self, path: Path, size: int) -> Optional[Dict]:
        """Check if file size violates policies."""
        relative_path = path.relative_to(self.root_dir)
        
        # Check if file is in allowed directory for large models
        is_in_allowed_dir = any(
            str(relative_path).startswith(allowed_dir) 
            for allowed_dir in self.ALLOWED_LARGE_MODEL_DIRS
        )
        
        # Determine violation level
        if size > self.SIZE_LIMITS['error'] and not is_in_allowed_dir:
            level = 'error'
            message = f"Model file exceeds maximum size limit ({self._format_size(self.SIZE_LIMITS['error'])})"
        elif size > self.SIZE_LIMITS['warning'] and not is_in_allowed_dir:
            level = 'warning'  
            message = f"Model file exceeds recommended size limit ({self._format_size(self.SIZE_LIMITS['warning'])})"
        elif size > self.SIZE_LIMITS['info']:
            level = 'info'
            message = f"Large model file detected"
        else:
            return None
        
        return {
            'level': level,
            'path': str(relative_path),
            'size': size,
            'size_formatted': self._format_size(size),
            'message': message,
            'recommendations': self._get_recommendations(size, is_in_allowed_dir)
        }
    
    def _format_size(self, size: int) -> str:
        """Format file size in human-readable format."""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size < 1024:
                return f"{size:.1f} {unit}"
            size /= 1024
        return f"{size:.1f} TB"
    
    def _get_recommendations(self, size: int, is_in_allowed_dir: bool) -> List[str]:
        """Get recommendations for handling large model files."""
        recommendations = []
        
        if size > self.SIZE_LIMITS['error']:
            recommendations.extend([
                "Use Git LFS (Large File Storage) for this model file",
                "Consider storing models in external storage (S3, GCS, etc.)",
                "Use model compression techniques to reduce file size",
                "Split large models into smaller chunks"
            ])
        elif size > self.SIZE_LIMITS['warning']:
            recommendations.extend([
                "Consider using Git LFS for better repository performance",
                "Evaluate if model compression can reduce file size",
                "Document why this large model is necessary in the repository"
            ])
        
        if not is_in_allowed_dir:
            recommendations.append(
                f"Move to an allowed directory: {', '.join(self.ALLOWED_LARGE_MODEL_DIRS)}"
            )
        
        return recommendations
    
    def generate_report(self, results: Dict) -> str:
        """Generate a detailed report of model size analysis."""
        report = ["# Model Size Analysis Report", ""]
        
        # Summary
        report.append(f"**Files Checked:** {results['checked_files']}")
        report.append(f"**Total Size:** {self._format_size(results['total_size'])}")
        report.append(f"**Violations:** {len(results['violations'])}")
        report.append(f"**Warnings:** {len(results['warnings'])}")
        report.append("")
        
        # Size limits
        report.append("## Size Limits")
        report.append(f"- **Hard Limit:** {self._format_size(self.SIZE_LIMITS['error'])}")
        report.append(f"- **Soft Limit:** {self._format_size(self.SIZE_LIMITS['warning'])}")
        report.append(f"- **Info Threshold:** {self._format_size(self.SIZE_LIMITS['info'])}")
        report.append("")
        
        # Violations
        if results['violations']:
            report.append("## ❌ Size Violations (Must Fix)")
            for violation in results['violations']:
                report.append(f"### {violation['path']}")
                report.append(f"- **Size:** {violation['size_formatted']}")
                report.append(f"- **Issue:** {violation['message']}")
                report.append("- **Recommendations:**")
                for rec in violation['recommendations']:
                    report.append(f"  - {rec}")
                report.append("")
        
        # Warnings  
        if results['warnings']:
            report.append("## ⚠️ Size Warnings (Should Review)")
            for warning in results['warnings']:
                report.append(f"### {warning['path']}")
                report.append(f"- **Size:** {warning['size_formatted']}")
                report.append(f"- **Issue:** {warning['message']}")
                report.append("- **Recommendations:**")
                for rec in warning['recommendations']:
                    report.append(f"  - {rec}")
                report.append("")
        
        # Info
        if results['info']:
            report.append("## ℹ️ Large Models (Informational)")
            for info in results['info']:
                report.append(f"- **{info['path']}:** {info['size_formatted']}")
            report.append("")
        
        # Git LFS setup instructions
        if results['violations'] or results['warnings']:
            report.append("## Git LFS Setup Instructions")
            report.append("```bash")
            report.append("# Install Git LFS")
            report.append("git lfs install")
            report.append("")
            report.append("# Track model files")
            report.extend([
                f"git lfs track '*{ext}'" for ext in sorted(self.MODEL_EXTENSIONS)
            ])
            report.append("")
            report.append("# Add .gitattributes and commit")
            report.append("git add .gitattributes")
            report.append("git commit -m 'Add Git LFS tracking for model files'")
            report.append("```")
        
        return "\n".join(report)


def main():
    """Main entry point for model size checking."""
    # Parse command line arguments
    scan_all = "--scan-all" in sys.argv
    generate_report = "--report" in sys.argv
    verbose = "--verbose" in sys.argv or "-v" in sys.argv
    
    # Get files to check from command line or stdin
    if len(sys.argv) > 1 and not any(arg.startswith('-') for arg in sys.argv[1:]):
        files_to_check = [arg for arg in sys.argv[1:] if not arg.startswith('-')]
    else:
        files_to_check = []
    
    # Initialize checker
    checker = ModelSizeChecker()
    
    # Check files
    if scan_all:
        print("🔍 Scanning entire repository for model files...")
        passed, results = checker.scan_repository()
    elif files_to_check:
        print(f"🔍 Checking {len(files_to_check)} specified files...")
        passed, results = checker.check_files(files_to_check)
    else:
        print("🔍 Checking files from git staged area...")
        # Get staged files
        try:
            import subprocess
            result = subprocess.run(
                ['git', 'diff', '--cached', '--name-only'],
                capture_output=True, text=True, check=True
            )
            staged_files = result.stdout.strip().split('\n') if result.stdout.strip() else []
            passed, results = checker.check_files(staged_files)
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("⚠️  Could not get staged files from git. Scanning repository...")
            passed, results = checker.scan_repository()
    
    # Print summary
    print(f"\n📊 Model Size Check Summary:")
    print(f"  Files checked: {results['checked_files']}")
    print(f"  Total size: {checker._format_size(results['total_size'])}")
    print(f"  Violations: {len(results['violations'])}")
    print(f"  Warnings: {len(results['warnings'])}")
    
    # Print violations
    if results['violations']:
        print(f"\n❌ {len(results['violations'])} size violation(s) found:")
        for violation in results['violations']:
            print(f"  - {violation['path']}: {violation['size_formatted']} ({violation['message']})")
    
    # Print warnings  
    if results['warnings'] and verbose:
        print(f"\n⚠️  {len(results['warnings'])} warning(s):")
        for warning in results['warnings']:
            print(f"  - {warning['path']}: {warning['size_formatted']}")
    
    # Generate detailed report
    if generate_report or results['violations']:
        report = checker.generate_report(results)
        with open("model_size_report.md", "w") as f:
            f.write(report)
        print(f"\n📝 Detailed report saved to model_size_report.md")
    
    # Exit with appropriate code
    if not passed:
        print(f"\n💡 Consider using Git LFS for large model files")
        print(f"   Run: git lfs track '*.pkl' '*.pth' '*.h5' etc.")
        sys.exit(1)
    else:
        print("\n✅ All model files are within size limits")


if __name__ == "__main__":
    main()