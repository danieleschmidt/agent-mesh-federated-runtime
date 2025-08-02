#!/usr/bin/env python3
"""
Repository automation script for Agent Mesh Federated Runtime.
Handles automated maintenance tasks, health checks, and repository optimization.
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import requests
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RepositoryAutomation:
    """Main repository automation orchestrator."""
    
    def __init__(self, repo_path: Path):
        self.repo_path = repo_path
        self.github_token = os.getenv('GITHUB_TOKEN')
        self.github_repo = os.getenv('GITHUB_REPOSITORY', 'danieleschmidt/agent-mesh-federated-runtime')
        self.api_url = 'https://api.github.com'
        
        # Initialize automation modules
        self.dependency_manager = DependencyManager(repo_path)
        self.code_quality = CodeQualityManager(repo_path)
        self.security_scanner = SecurityScanner(repo_path)
        self.documentation_manager = DocumentationManager(repo_path)
        self.issue_manager = IssueManager(self.github_token, self.github_repo)
    
    async def run_automated_maintenance(self, tasks: List[str] = None) -> Dict:
        """Run automated maintenance tasks."""
        if tasks is None:
            tasks = [
                'dependency_check',
                'security_scan',
                'code_quality_check',
                'documentation_update',
                'issue_triage',
                'cleanup'
            ]
        
        results = {
            'timestamp': datetime.utcnow().isoformat(),
            'tasks': {},
            'summary': {
                'total_tasks': len(tasks),
                'successful': 0,
                'failed': 0,
                'warnings': 0
            }
        }
        
        logger.info(f"Starting automated maintenance with {len(tasks)} tasks...")
        
        for task in tasks:
            logger.info(f"Executing task: {task}")
            start_time = time.time()
            
            try:
                if task == 'dependency_check':
                    result = await self.dependency_manager.check_and_update()
                elif task == 'security_scan':
                    result = await self.security_scanner.comprehensive_scan()
                elif task == 'code_quality_check':
                    result = await self.code_quality.analyze_and_improve()
                elif task == 'documentation_update':
                    result = await self.documentation_manager.update_docs()
                elif task == 'issue_triage':
                    result = await self.issue_manager.triage_issues()
                elif task == 'cleanup':
                    result = await self._cleanup_repository()
                else:
                    result = {'status': 'skipped', 'message': f'Unknown task: {task}'}
                
                results['tasks'][task] = {
                    **result,
                    'duration': time.time() - start_time
                }
                
                if result.get('status') == 'success':
                    results['summary']['successful'] += 1
                elif result.get('status') == 'warning':
                    results['summary']['warnings'] += 1
                else:
                    results['summary']['failed'] += 1
                
                logger.info(f"Task {task} completed: {result.get('status', 'unknown')}")
                
            except Exception as e:
                logger.error(f"Task {task} failed: {e}")
                results['tasks'][task] = {
                    'status': 'error',
                    'error': str(e),
                    'duration': time.time() - start_time
                }
                results['summary']['failed'] += 1
        
        # Generate maintenance report
        await self._generate_maintenance_report(results)
        
        logger.info(f"Automated maintenance completed: {results['summary']}")
        return results
    
    async def _cleanup_repository(self) -> Dict:
        """Clean up repository artifacts and temporary files."""
        cleanup_tasks = []
        
        try:
            # Clean Python cache
            cleanup_tasks.append(self._run_command(['find', '.', '-name', '__pycache__', '-type', 'd', '-exec', 'rm', '-rf', '{}', '+']))
            cleanup_tasks.append(self._run_command(['find', '.', '-name', '*.pyc', '-delete']))
            
            # Clean build artifacts
            cleanup_tasks.append(self._run_command(['rm', '-rf', 'build/', 'dist/', '*.egg-info/']))
            
            # Clean test artifacts
            cleanup_tasks.append(self._run_command(['rm', '-rf', '.pytest_cache/', '.coverage', 'htmlcov/']))
            
            # Clean Docker artifacts (if safe)
            if self._is_safe_to_clean_docker():
                cleanup_tasks.append(self._run_command(['docker', 'system', 'prune', '-f']))
            
            # Clean old log files
            log_files = list(Path('.').glob('**/*.log'))
            old_logs = [f for f in log_files if self._is_old_file(f, days=7)]
            if old_logs:
                cleanup_tasks.append(self._run_command(['rm'] + [str(f) for f in old_logs]))
            
            successful_tasks = sum(1 for task in cleanup_tasks if task.returncode == 0)
            
            return {
                'status': 'success',
                'message': f'Cleanup completed: {successful_tasks}/{len(cleanup_tasks)} tasks successful',
                'cleaned_items': len(cleanup_tasks)
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Cleanup failed: {e}',
                'error': str(e)
            }
    
    def _run_command(self, cmd: List[str], cwd: Path = None) -> subprocess.CompletedProcess:
        """Run shell command safely."""
        try:
            return subprocess.run(
                cmd,
                cwd=cwd or self.repo_path,
                capture_output=True,
                text=True,
                timeout=300
            )
        except subprocess.TimeoutExpired:
            logger.warning(f"Command timed out: {' '.join(cmd)}")
            return subprocess.CompletedProcess(cmd, 1, '', 'Command timed out')
        except Exception as e:
            logger.error(f"Command failed: {' '.join(cmd)}: {e}")
            return subprocess.CompletedProcess(cmd, 1, '', str(e))
    
    def _is_safe_to_clean_docker(self) -> bool:
        """Check if it's safe to clean Docker artifacts."""
        # Only clean if we're in a CI environment or explicitly allowed
        return os.getenv('CI') == 'true' or os.getenv('ALLOW_DOCKER_CLEANUP') == 'true'
    
    def _is_old_file(self, file_path: Path, days: int) -> bool:
        """Check if file is older than specified days."""
        try:
            file_age = datetime.now() - datetime.fromtimestamp(file_path.stat().st_mtime)
            return file_age > timedelta(days=days)
        except Exception:
            return False
    
    async def _generate_maintenance_report(self, results: Dict):
        """Generate and save maintenance report."""
        try:
            report_path = self.repo_path / '.github' / 'maintenance-report.json'
            
            # Load previous reports
            reports = []
            if report_path.exists():
                with open(report_path) as f:
                    reports = json.load(f)
            
            # Add current report
            reports.append(results)
            
            # Keep only last 30 reports
            reports = reports[-30:]
            
            # Save updated reports
            with open(report_path, 'w') as f:
                json.dump(reports, f, indent=2)
            
            logger.info(f"Maintenance report saved to {report_path}")
            
        except Exception as e:
            logger.error(f"Failed to save maintenance report: {e}")


class DependencyManager:
    """Manages project dependencies and updates."""
    
    def __init__(self, repo_path: Path):
        self.repo_path = repo_path
    
    async def check_and_update(self) -> Dict:
        """Check and update dependencies."""
        results = {
            'python_updates': [],
            'node_updates': [],
            'security_fixes': [],
            'status': 'success'
        }
        
        try:
            # Check Python dependencies
            if (self.repo_path / 'requirements.txt').exists():
                python_result = await self._check_python_dependencies()
                results['python_updates'] = python_result
            
            # Check Node.js dependencies
            if (self.repo_path / 'package.json').exists():
                node_result = await self._check_node_dependencies()
                results['node_updates'] = node_result
            
            # Check for security vulnerabilities
            security_result = await self._check_security_vulnerabilities()
            results['security_fixes'] = security_result
            
            total_updates = len(results['python_updates']) + len(results['node_updates']) + len(results['security_fixes'])
            
            if total_updates > 0:
                results['status'] = 'warning'
                results['message'] = f'Found {total_updates} dependency updates available'
            else:
                results['message'] = 'All dependencies are up to date'
            
        except Exception as e:
            results['status'] = 'error'
            results['error'] = str(e)
        
        return results
    
    async def _check_python_dependencies(self) -> List[Dict]:
        """Check Python dependencies for updates."""
        updates = []
        
        try:
            # Use pip-outdated or similar tool
            result = subprocess.run(
                ['pip', 'list', '--outdated', '--format=json'],
                capture_output=True,
                text=True,
                cwd=self.repo_path
            )
            
            if result.returncode == 0:
                outdated = json.loads(result.stdout)
                for package in outdated:
                    updates.append({
                        'name': package['name'],
                        'current': package['version'],
                        'latest': package['latest_version'],
                        'type': 'python'
                    })
        
        except Exception as e:
            logger.warning(f"Failed to check Python dependencies: {e}")
        
        return updates
    
    async def _check_node_dependencies(self) -> List[Dict]:
        """Check Node.js dependencies for updates."""
        updates = []
        
        try:
            result = subprocess.run(
                ['npm', 'outdated', '--json'],
                capture_output=True,
                text=True,
                cwd=self.repo_path
            )
            
            if result.stdout:
                outdated = json.loads(result.stdout)
                for package, info in outdated.items():
                    updates.append({
                        'name': package,
                        'current': info['current'],
                        'latest': info['latest'],
                        'type': 'nodejs'
                    })
        
        except Exception as e:
            logger.warning(f"Failed to check Node.js dependencies: {e}")
        
        return updates
    
    async def _check_security_vulnerabilities(self) -> List[Dict]:
        """Check for security vulnerabilities in dependencies."""
        vulnerabilities = []
        
        try:
            # Python security check
            result = subprocess.run(
                ['safety', 'check', '--json'],
                capture_output=True,
                text=True,
                cwd=self.repo_path
            )
            
            if result.stdout:
                safety_report = json.loads(result.stdout)
                for vuln in safety_report:
                    vulnerabilities.append({
                        'package': vuln['package_name'],
                        'vulnerability': vuln['vulnerability_id'],
                        'severity': 'high',  # Safety reports are typically high severity
                        'type': 'python_security'
                    })
        
        except Exception as e:
            logger.warning(f"Failed to check Python security: {e}")
        
        try:
            # Node.js security check
            result = subprocess.run(
                ['npm', 'audit', '--json'],
                capture_output=True,
                text=True,
                cwd=self.repo_path
            )
            
            if result.stdout:
                audit_report = json.loads(result.stdout)
                vulnerabilities_data = audit_report.get('vulnerabilities', {})
                
                for package, vuln_info in vulnerabilities_data.items():
                    vulnerabilities.append({
                        'package': package,
                        'severity': vuln_info.get('severity', 'unknown'),
                        'type': 'nodejs_security',
                        'via': vuln_info.get('via', [])
                    })
        
        except Exception as e:
            logger.warning(f"Failed to check Node.js security: {e}")
        
        return vulnerabilities


class CodeQualityManager:
    """Manages code quality analysis and improvements."""
    
    def __init__(self, repo_path: Path):
        self.repo_path = repo_path
    
    async def analyze_and_improve(self) -> Dict:
        """Analyze code quality and suggest improvements."""
        results = {
            'linting_issues': 0,
            'formatting_issues': 0,
            'complexity_issues': 0,
            'suggestions': [],
            'status': 'success'
        }
        
        try:
            # Run linting
            lint_result = await self._run_linting()
            results['linting_issues'] = lint_result['issues']
            results['suggestions'].extend(lint_result['suggestions'])
            
            # Check formatting
            format_result = await self._check_formatting()
            results['formatting_issues'] = format_result['issues']
            results['suggestions'].extend(format_result['suggestions'])
            
            # Analyze complexity
            complexity_result = await self._analyze_complexity()
            results['complexity_issues'] = complexity_result['issues']
            results['suggestions'].extend(complexity_result['suggestions'])
            
            total_issues = results['linting_issues'] + results['formatting_issues'] + results['complexity_issues']
            
            if total_issues > 50:
                results['status'] = 'warning'
                results['message'] = f'Found {total_issues} code quality issues'
            elif total_issues > 0:
                results['status'] = 'info'
                results['message'] = f'Found {total_issues} minor code quality issues'
            else:
                results['message'] = 'Code quality is good'
            
        except Exception as e:
            results['status'] = 'error'
            results['error'] = str(e)
        
        return results
    
    async def _run_linting(self) -> Dict:
        """Run code linting."""
        issues = 0
        suggestions = []
        
        try:
            # Python linting with flake8
            result = subprocess.run(
                ['flake8', 'src/', 'tests/', '--count', '--statistics'],
                capture_output=True,
                text=True,
                cwd=self.repo_path
            )
            
            if result.stdout:
                lines = result.stdout.strip().split('\n')
                for line in lines:
                    if line and line[0].isdigit():
                        count = int(line.split()[0])
                        issues += count
                        suggestions.append(f"Fix {count} linting issues: {line}")
        
        except Exception as e:
            logger.warning(f"Linting check failed: {e}")
        
        return {'issues': issues, 'suggestions': suggestions}
    
    async def _check_formatting(self) -> Dict:
        """Check code formatting."""
        issues = 0
        suggestions = []
        
        try:
            # Check Python formatting with black
            result = subprocess.run(
                ['black', '--check', '--diff', 'src/', 'tests/'],
                capture_output=True,
                text=True,
                cwd=self.repo_path
            )
            
            if result.returncode != 0:
                diff_lines = len(result.stdout.split('\n'))
                issues = diff_lines
                suggestions.append(f"Run 'black src/ tests/' to fix {issues} formatting issues")
        
        except Exception as e:
            logger.warning(f"Formatting check failed: {e}")
        
        return {'issues': issues, 'suggestions': suggestions}
    
    async def _analyze_complexity(self) -> Dict:
        """Analyze code complexity."""
        issues = 0
        suggestions = []
        
        try:
            # Use radon for complexity analysis
            result = subprocess.run(
                ['radon', 'cc', 'src/', '--min', 'C'],
                capture_output=True,
                text=True,
                cwd=self.repo_path
            )
            
            if result.stdout:
                complex_functions = len([line for line in result.stdout.split('\n') 
                                       if line.strip() and not line.startswith('=')])
                issues = complex_functions
                if issues > 0:
                    suggestions.append(f"Consider refactoring {issues} complex functions")
        
        except Exception as e:
            logger.warning(f"Complexity analysis failed: {e}")
        
        return {'issues': issues, 'suggestions': suggestions}


class SecurityScanner:
    """Handles security scanning and vulnerability detection."""
    
    def __init__(self, repo_path: Path):
        self.repo_path = repo_path
    
    async def comprehensive_scan(self) -> Dict:
        """Run comprehensive security scan."""
        results = {
            'vulnerabilities': [],
            'security_hotspots': [],
            'compliance_issues': [],
            'recommendations': [],
            'status': 'success'
        }
        
        try:
            # Run Bandit for Python security issues
            bandit_result = await self._run_bandit_scan()
            results['vulnerabilities'].extend(bandit_result['vulnerabilities'])
            results['recommendations'].extend(bandit_result['recommendations'])
            
            # Check dependency vulnerabilities
            deps_result = await self._check_dependency_vulnerabilities()
            results['vulnerabilities'].extend(deps_result['vulnerabilities'])
            
            # Check for secrets in code
            secrets_result = await self._check_for_secrets()
            results['security_hotspots'].extend(secrets_result['hotspots'])
            
            total_issues = len(results['vulnerabilities']) + len(results['security_hotspots'])
            
            if total_issues > 0:
                results['status'] = 'warning'
                results['message'] = f'Found {total_issues} security issues'
            else:
                results['message'] = 'No security issues found'
            
        except Exception as e:
            results['status'] = 'error'
            results['error'] = str(e)
        
        return results
    
    async def _run_bandit_scan(self) -> Dict:
        """Run Bandit security scan."""
        vulnerabilities = []
        recommendations = []
        
        try:
            result = subprocess.run(
                ['bandit', '-r', 'src/', '-f', 'json'],
                capture_output=True,
                text=True,
                cwd=self.repo_path
            )
            
            if result.stdout:
                bandit_report = json.loads(result.stdout)
                for issue in bandit_report.get('results', []):
                    vulnerabilities.append({
                        'type': 'code_security',
                        'severity': issue['issue_severity'],
                        'confidence': issue['issue_confidence'],
                        'description': issue['issue_text'],
                        'file': issue['filename'],
                        'line': issue['line_number']
                    })
                
                if vulnerabilities:
                    recommendations.append("Review and fix security issues identified by Bandit")
        
        except Exception as e:
            logger.warning(f"Bandit scan failed: {e}")
        
        return {'vulnerabilities': vulnerabilities, 'recommendations': recommendations}
    
    async def _check_dependency_vulnerabilities(self) -> Dict:
        """Check for dependency vulnerabilities."""
        vulnerabilities = []
        
        try:
            # This would typically use the dependency manager results
            # For now, we'll do a basic check
            result = subprocess.run(
                ['safety', 'check', '--json'],
                capture_output=True,
                text=True,
                cwd=self.repo_path
            )
            
            if result.stdout and result.stdout.strip() != '[]':
                safety_issues = json.loads(result.stdout)
                for issue in safety_issues:
                    vulnerabilities.append({
                        'type': 'dependency_vulnerability',
                        'package': issue.get('package_name'),
                        'severity': 'high',
                        'description': issue.get('advisory'),
                        'affected_versions': issue.get('affected_versions')
                    })
        
        except Exception as e:
            logger.warning(f"Dependency vulnerability check failed: {e}")
        
        return {'vulnerabilities': vulnerabilities}
    
    async def _check_for_secrets(self) -> Dict:
        """Check for exposed secrets in code."""
        hotspots = []
        
        try:
            # Use detect-secrets if available
            result = subprocess.run(
                ['detect-secrets', 'scan', '--all-files'],
                capture_output=True,
                text=True,
                cwd=self.repo_path
            )
            
            if result.stdout:
                secrets_data = json.loads(result.stdout)
                for filename, secrets in secrets_data.get('results', {}).items():
                    for secret in secrets:
                        hotspots.append({
                            'type': 'potential_secret',
                            'file': filename,
                            'line': secret.get('line_number'),
                            'description': f"Potential {secret.get('type')} secret detected"
                        })
        
        except Exception as e:
            logger.warning(f"Secret detection failed: {e}")
        
        return {'hotspots': hotspots}


class DocumentationManager:
    """Manages documentation updates and maintenance."""
    
    def __init__(self, repo_path: Path):
        self.repo_path = repo_path
    
    async def update_docs(self) -> Dict:
        """Update documentation automatically."""
        results = {
            'updated_files': [],
            'broken_links': [],
            'outdated_sections': [],
            'status': 'success'
        }
        
        try:
            # Check for outdated API documentation
            api_result = await self._check_api_documentation()
            results['outdated_sections'].extend(api_result['outdated'])
            results['updated_files'].extend(api_result['updated'])
            
            # Check for broken links
            links_result = await self._check_documentation_links()
            results['broken_links'].extend(links_result['broken'])
            
            # Update version references
            version_result = await self._update_version_references()
            results['updated_files'].extend(version_result['updated'])
            
            total_issues = len(results['broken_links']) + len(results['outdated_sections'])
            if total_issues > 0:
                results['status'] = 'warning'
                results['message'] = f'Found {total_issues} documentation issues'
            else:
                results['message'] = 'Documentation is up to date'
            
        except Exception as e:
            results['status'] = 'error'
            results['error'] = str(e)
        
        return results
    
    async def _check_api_documentation(self) -> Dict:
        """Check API documentation for outdated content."""
        outdated = []
        updated = []
        
        # This would compare actual API with documented API
        # For now, we'll do a basic check
        api_docs = list(self.repo_path.glob('docs/api/**/*.md'))
        
        for doc in api_docs:
            try:
                with open(doc) as f:
                    content = f.read()
                
                # Check for common outdated patterns
                if 'TODO' in content or 'FIXME' in content:
                    outdated.append(str(doc.relative_to(self.repo_path)))
            
            except Exception as e:
                logger.warning(f"Failed to check {doc}: {e}")
        
        return {'outdated': outdated, 'updated': updated}
    
    async def _check_documentation_links(self) -> Dict:
        """Check for broken links in documentation."""
        broken = []
        
        md_files = list(self.repo_path.glob('**/*.md'))
        
        for md_file in md_files:
            try:
                with open(md_file) as f:
                    content = f.read()
                
                # Basic link checking (would need more sophisticated implementation)
                import re
                links = re.findall(r'\[([^\]]+)\]\(([^)]+)\)', content)
                
                for link_text, link_url in links:
                    if link_url.startswith('http'):
                        # Would check if HTTP links are accessible
                        pass
                    elif link_url.startswith('/') or not link_url.startswith('#'):
                        # Check local file links
                        if link_url.startswith('/'):
                            target = self.repo_path / link_url[1:]
                        else:
                            target = md_file.parent / link_url
                        
                        if not target.exists():
                            broken.append({
                                'file': str(md_file.relative_to(self.repo_path)),
                                'link': link_url,
                                'text': link_text
                            })
            
            except Exception as e:
                logger.warning(f"Failed to check links in {md_file}: {e}")
        
        return {'broken': broken}
    
    async def _update_version_references(self) -> Dict:
        """Update version references in documentation."""
        updated = []
        
        # This would update version numbers in documentation
        # Implementation would depend on versioning strategy
        
        return {'updated': updated}


class IssueManager:
    """Manages GitHub issues and automated triage."""
    
    def __init__(self, github_token: str, repo: str):
        self.token = github_token
        self.repo = repo
        self.api_url = 'https://api.github.com'
    
    async def triage_issues(self) -> Dict:
        """Automatically triage GitHub issues."""
        results = {
            'processed_issues': 0,
            'labeled_issues': 0,
            'assigned_issues': 0,
            'closed_stale_issues': 0,
            'status': 'success'
        }
        
        if not self.token:
            return {
                'status': 'skipped',
                'message': 'GitHub token not available for issue triage'
            }
        
        try:
            headers = {'Authorization': f'token {self.token}'}
            
            # Get open issues
            issues_response = requests.get(
                f"{self.api_url}/repos/{self.repo}/issues",
                headers=headers,
                params={'state': 'open', 'per_page': 100}
            )
            issues_response.raise_for_status()
            issues = issues_response.json()
            
            for issue in issues:
                if 'pull_request' in issue:
                    continue  # Skip pull requests
                
                results['processed_issues'] += 1
                
                # Auto-label based on title/body content
                if await self._auto_label_issue(issue, headers):
                    results['labeled_issues'] += 1
                
                # Check if issue is stale
                if await self._check_stale_issue(issue, headers):
                    results['closed_stale_issues'] += 1
            
            results['message'] = f"Processed {results['processed_issues']} issues"
            
        except Exception as e:
            results['status'] = 'error'
            results['error'] = str(e)
        
        return results
    
    async def _auto_label_issue(self, issue: Dict, headers: Dict) -> bool:
        """Automatically label issue based on content."""
        title = issue.get('title', '').lower()
        body = issue.get('body', '').lower()
        current_labels = [label['name'] for label in issue.get('labels', [])]
        
        labels_to_add = []
        
        # Bug detection
        if any(word in title or word in body for word in ['bug', 'error', 'exception', 'crash', 'fail']):
            if 'bug' not in current_labels:
                labels_to_add.append('bug')
        
        # Feature request detection
        if any(word in title or word in body for word in ['feature', 'enhancement', 'improve', 'add']):
            if 'enhancement' not in current_labels:
                labels_to_add.append('enhancement')
        
        # Documentation issues
        if any(word in title or word in body for word in ['documentation', 'docs', 'readme']):
            if 'documentation' not in current_labels:
                labels_to_add.append('documentation')
        
        # Priority detection
        if any(word in title or word in body for word in ['urgent', 'critical', 'blocker']):
            if 'priority: high' not in current_labels:
                labels_to_add.append('priority: high')
        
        if labels_to_add:
            try:
                # Add labels to issue
                response = requests.post(
                    f"{self.api_url}/repos/{self.repo}/issues/{issue['number']}/labels",
                    headers=headers,
                    json={'labels': labels_to_add}
                )
                response.raise_for_status()
                return True
            except Exception as e:
                logger.warning(f"Failed to add labels to issue #{issue['number']}: {e}")
        
        return False
    
    async def _check_stale_issue(self, issue: Dict, headers: Dict) -> bool:
        """Check if issue is stale and should be closed."""
        updated_at = datetime.fromisoformat(issue['updated_at'].replace('Z', '+00:00'))
        now = datetime.now(updated_at.tzinfo)
        days_since_update = (now - updated_at).days
        
        # Close issues that haven't been updated in 90 days and have no recent activity
        if days_since_update > 90:
            current_labels = [label['name'] for label in issue.get('labels', [])]
            
            # Don't close issues with certain labels
            protected_labels = ['pinned', 'feature', 'epic', 'roadmap']
            if not any(label in current_labels for label in protected_labels):
                try:
                    # Add stale label and close
                    requests.post(
                        f"{self.api_url}/repos/{self.repo}/issues/{issue['number']}/labels",
                        headers=headers,
                        json={'labels': ['stale']}
                    )
                    
                    # Close the issue
                    requests.patch(
                        f"{self.api_url}/repos/{self.repo}/issues/{issue['number']}",
                        headers=headers,
                        json={
                            'state': 'closed',
                            'state_reason': 'not_planned'
                        }
                    )
                    
                    # Add closing comment
                    requests.post(
                        f"{self.api_url}/repos/{self.repo}/issues/{issue['number']}/comments",
                        headers=headers,
                        json={
                            'body': 'This issue has been automatically closed due to inactivity. If this issue is still relevant, please reopen it or create a new issue.'
                        }
                    )
                    
                    return True
                    
                except Exception as e:
                    logger.warning(f"Failed to close stale issue #{issue['number']}: {e}")
        
        return False


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Repository automation for Agent Mesh')
    parser.add_argument('--tasks', nargs='+', 
                       choices=['dependency_check', 'security_scan', 'code_quality_check', 
                               'documentation_update', 'issue_triage', 'cleanup'],
                       help='Specific tasks to run')
    parser.add_argument('--repo-path', default='.', help='Path to repository')
    parser.add_argument('--output', help='Output file for results')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    repo_path = Path(args.repo_path).resolve()
    automation = RepositoryAutomation(repo_path)
    
    try:
        results = await automation.run_automated_maintenance(args.tasks)
        
        # Output results
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Results written to {args.output}")
        else:
            print(json.dumps(results, indent=2))
        
        # Exit with non-zero code if any tasks failed
        if results['summary']['failed'] > 0:
            sys.exit(1)
        
    except Exception as e:
        logger.error(f"Automation failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    import asyncio
    asyncio.run(main())