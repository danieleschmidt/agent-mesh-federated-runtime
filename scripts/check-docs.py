#!/usr/bin/env python3
"""
Documentation checker for Agent Mesh Federated Runtime.
Validates documentation completeness and quality.
"""

import argparse
import ast
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


class DocstringChecker:
    """Checks Python docstrings for completeness and quality."""
    
    def __init__(self, root_dir: Path):
        self.root_dir = root_dir
        self.issues = []
    
    def check_file(self, file_path: Path) -> List[Dict]:
        """Check a single Python file for docstring issues."""
        file_issues = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content, filename=str(file_path))
            
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    issue = self._check_node_docstring(node, file_path)
                    if issue:
                        file_issues.append(issue)
        
        except Exception as e:
            file_issues.append({
                'file': str(file_path.relative_to(self.root_dir)),
                'type': 'parse_error',
                'message': f"Could not parse file: {e}",
                'line': 1
            })
        
        return file_issues
    
    def _check_node_docstring(self, node, file_path: Path) -> Optional[Dict]:
        """Check if a function/class has proper docstring."""
        # Skip private methods and test methods
        if node.name.startswith('_') and not node.name.startswith('__'):
            return None
        
        if node.name.startswith('test_'):
            return None
        
        docstring = ast.get_docstring(node)
        
        if not docstring:
            return {
                'file': str(file_path.relative_to(self.root_dir)),
                'type': 'missing_docstring',
                'name': node.name,
                'kind': type(node).__name__.replace('Def', '').lower(),
                'message': f"Missing docstring for {type(node).__name__.replace('Def', '').lower()} '{node.name}'",
                'line': node.lineno
            }
        
        # Check docstring quality
        quality_issues = self._check_docstring_quality(docstring, node, file_path)
        return quality_issues
    
    def _check_docstring_quality(self, docstring: str, node, file_path: Path) -> Optional[Dict]:
        """Check docstring quality and completeness."""
        issues = []
        
        # Check minimum length
        if len(docstring.strip()) < 10:
            issues.append("Docstring too short")
        
        # Check for proper formatting (Google style)
        if isinstance(node, ast.FunctionDef) and node.args.args:
            args = [arg.arg for arg in node.args.args if arg.arg != 'self']
            if args and 'Args:' not in docstring:
                issues.append("Missing Args section")
        
        # Check for return annotation and docstring
        if (isinstance(node, ast.FunctionDef) and 
            node.returns and 
            'Returns:' not in docstring and
            'return' not in docstring.lower()):
            issues.append("Missing Returns section")
        
        if issues:
            return {
                'file': str(file_path.relative_to(self.root_dir)),
                'type': 'docstring_quality',
                'name': node.name,
                'kind': type(node).__name__.replace('Def', '').lower(),
                'message': f"Docstring quality issues: {', '.join(issues)}",
                'line': node.lineno
            }
        
        return None


class ReadmeChecker:
    """Checks README and documentation files for completeness."""
    
    def __init__(self, root_dir: Path):
        self.root_dir = root_dir
        self.required_sections = [
            'installation',
            'usage',
            'api',
            'contributing',
            'license'
        ]
    
    def check_readme(self) -> List[Dict]:
        """Check README.md for required sections."""
        issues = []
        readme_path = self.root_dir / "README.md"
        
        if not readme_path.exists():
            return [{
                'file': 'README.md',
                'type': 'missing_file',
                'message': 'README.md file is missing',
                'line': 1
            }]
        
        try:
            with open(readme_path, 'r', encoding='utf-8') as f:
                content = f.read().lower()
            
            for section in self.required_sections:
                if section not in content:
                    issues.append({
                        'file': 'README.md',
                        'type': 'missing_section',
                        'message': f"Missing required section: {section}",
                        'section': section,
                        'line': 1
                    })
            
            # Check for broken links
            link_pattern = r'\[([^\]]+)\]\(([^)]+)\)'
            links = re.findall(link_pattern, content)
            
            for link_text, link_url in links:
                if link_url.startswith('http'):
                    continue  # Skip external links for now
                
                if link_url.startswith('#'):
                    continue  # Skip anchor links for now
                
                # Check local file links
                link_path = self.root_dir / link_url
                if not link_path.exists():
                    issues.append({
                        'file': 'README.md',
                        'type': 'broken_link',
                        'message': f"Broken link: {link_url}",
                        'link': link_url,
                        'line': 1
                    })
        
        except Exception as e:
            issues.append({
                'file': 'README.md',
                'type': 'read_error',
                'message': f"Could not read README.md: {e}",
                'line': 1
            })
        
        return issues


class ApiDocChecker:
    """Checks API documentation completeness."""
    
    def __init__(self, root_dir: Path):
        self.root_dir = root_dir
        self.src_dir = root_dir / "src"
    
    def check_api_docs(self) -> List[Dict]:
        """Check if all public modules have API documentation."""
        issues = []
        
        if not self.src_dir.exists():
            return issues
        
        # Find all Python modules
        python_files = list(self.src_dir.rglob("*.py"))
        public_modules = []
        
        for py_file in python_files:
            if '__pycache__' in str(py_file):
                continue
            if py_file.name.startswith('_') and py_file.name != '__init__.py':
                continue
            if 'test' in py_file.name:
                continue
            
            public_modules.append(py_file)
        
        # Check if docs/api directory exists
        api_docs_dir = self.root_dir / "docs" / "api"
        if not api_docs_dir.exists():
            issues.append({
                'file': 'docs/api/',
                'type': 'missing_directory',
                'message': 'API documentation directory missing',
                'line': 1
            })
            return issues
        
        # Check for API documentation files
        for module_file in public_modules:
            relative_path = module_file.relative_to(self.src_dir)
            expected_doc = api_docs_dir / f"{relative_path.stem}.md"
            
            if not expected_doc.exists():
                issues.append({
                    'file': str(relative_path),
                    'type': 'missing_api_doc',
                    'message': f"Missing API documentation for {relative_path}",
                    'expected_doc': str(expected_doc.relative_to(self.root_dir)),
                    'line': 1
                })
        
        return issues


def find_python_files(root_dir: Path) -> List[Path]:
    """Find all Python files to check."""
    python_files = []
    
    for file_path in root_dir.rglob("*.py"):
        # Skip excluded directories
        if any(exclude in file_path.parts for exclude in ['venv', '.venv', 'node_modules', '__pycache__', 'build', 'dist']):
            continue
        
        # Skip generated files
        if file_path.name.endswith('_pb2.py') or file_path.name.endswith('_pb2_grpc.py'):
            continue
        
        python_files.append(file_path)
    
    return python_files


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Check documentation completeness and quality"
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Root directory to scan (default: current directory)"
    )
    parser.add_argument(
        "--check-docstrings",
        action="store_true",
        help="Check Python docstrings"
    )
    parser.add_argument(
        "--check-readme",
        action="store_true", 
        help="Check README completeness"
    )
    parser.add_argument(
        "--check-api",
        action="store_true",
        help="Check API documentation"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all documentation checks"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output"
    )
    
    args = parser.parse_args()
    
    if not any([args.check_docstrings, args.check_readme, args.check_api, args.all]):
        args.all = True
    
    all_issues = []
    
    print(f"📖 Checking documentation in: {args.root}")
    
    # Check docstrings
    if args.check_docstrings or args.all:
        print("🐍 Checking Python docstrings...")
        docstring_checker = DocstringChecker(args.root)
        python_files = find_python_files(args.root)
        
        for py_file in python_files:
            issues = docstring_checker.check_file(py_file)
            all_issues.extend(issues)
        
        if args.verbose:
            print(f"  Checked {len(python_files)} Python files")
    
    # Check README
    if args.check_readme or args.all:
        print("📄 Checking README completeness...")
        readme_checker = ReadmeChecker(args.root)
        readme_issues = readme_checker.check_readme()
        all_issues.extend(readme_issues)
    
    # Check API documentation
    if args.check_api or args.all:
        print("📚 Checking API documentation...")
        api_checker = ApiDocChecker(args.root)
        api_issues = api_checker.check_api_docs()
        all_issues.extend(api_issues)
    
    # Report results
    if not all_issues:
        print("✅ Documentation checks passed!")
        return 0
    
    print(f"\n❌ Found {len(all_issues)} documentation issues:")
    
    # Group issues by type
    issues_by_type = {}
    for issue in all_issues:
        issue_type = issue['type']
        if issue_type not in issues_by_type:
            issues_by_type[issue_type] = []
        issues_by_type[issue_type].append(issue)
    
    for issue_type, issues in issues_by_type.items():
        print(f"\n{issue_type.replace('_', ' ').title()} ({len(issues)}):")
        for issue in issues:
            file_info = f"  {issue['file']}"
            if 'line' in issue:
                file_info += f":{issue['line']}"
            if 'name' in issue:
                file_info += f" ({issue['name']})"
            print(f"{file_info}: {issue['message']}")
    
    return 1


if __name__ == "__main__":
    sys.exit(main())