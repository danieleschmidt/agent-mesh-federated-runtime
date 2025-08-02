#!/usr/bin/env python3
"""
Terragon Autonomous SDLC Value Discovery Scoring Engine
Advanced Repository Optimization Focus
"""

import json
import yaml
import subprocess
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import re
import statistics

@dataclass
class WorkItem:
    id: str
    title: str
    description: str
    category: str  # security, performance, technical_debt, feature, infrastructure, documentation
    type: str     # specific type within category
    files_affected: List[str]
    estimated_effort: float  # hours
    priority: str  # critical, high, medium, low
    
    # WSJF Components
    user_business_value: int  # 1-10
    time_criticality: int     # 1-10
    risk_reduction: int       # 1-10
    opportunity_enablement: int # 1-10
    job_size: float          # story points
    
    # ICE Components  
    impact: int              # 1-10
    confidence: int          # 1-10
    ease: int               # 1-10
    
    # Technical Debt Specific
    debt_impact: float       # maintenance hours saved
    debt_interest: float     # future cost growth
    hotspot_multiplier: float # 1-5x based on churn
    
    # Risk and Security
    security_severity: str   # critical, high, medium, low, none
    compliance_impact: bool
    breaking_change_risk: float # 0-1
    
    # Computed Scores
    wsjf_score: float = 0
    ice_score: float = 0
    technical_debt_score: float = 0
    composite_score: float = 0

class ValueDiscoveryEngine:
    def __init__(self, config_path: str = ".terragon/config.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.metrics_path = Path(".terragon/value-metrics.json")
        self.metrics = self._load_metrics()
        
    def _load_config(self) -> Dict:
        with open(self.config_path) as f:
            return yaml.safe_load(f)
    
    def _load_metrics(self) -> Dict:
        if self.metrics_path.exists():
            with open(self.metrics_path) as f:
                return json.load(f)
        return {}
    
    def _save_metrics(self):
        with open(self.metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
    
    def discover_work_items(self) -> List[WorkItem]:
        """Comprehensive signal harvesting from multiple sources"""
        items = []
        
        # 1. Git History Analysis
        items.extend(self._discover_from_git_history())
        
        # 2. Static Analysis
        items.extend(self._discover_from_static_analysis())
        
        # 3. Security Scanning
        items.extend(self._discover_from_security_scans())
        
        # 4. Performance Monitoring
        items.extend(self._discover_from_performance_data())
        
        # 5. Dependency Analysis
        items.extend(self._discover_from_dependencies())
        
        # 6. Configuration Analysis
        items.extend(self._discover_from_configurations())
        
        return items
    
    def _discover_from_git_history(self) -> List[WorkItem]:
        """Extract TODOs, FIXMEs, and technical debt markers from Git history"""
        items = []
        
        try:
            # Search for debt markers in current codebase
            result = subprocess.run([
                'grep', '-r', '-n', '-i', 
                '--include=*.py', '--include=*.js', '--include=*.md',
                '-E', '(TODO|FIXME|HACK|XXX|DEPRECATED|TEMP)',
                'src/', 'tests/', 'docs/'
            ], capture_output=True, text=True, cwd='.')
            
            for line in result.stdout.split('\n'):
                if line.strip():
                    match = re.match(r'([^:]+):(\d+):(.+)', line)
                    if match:
                        file_path, line_num, content = match.groups()
                        
                        # Extract debt type and severity
                        debt_type = self._extract_debt_type(content)
                        severity = self._assess_debt_severity(content, file_path)
                        
                        items.append(WorkItem(
                            id=f"debt-{len(items)+1:03d}",
                            title=f"Address {debt_type} in {file_path}:{line_num}",
                            description=content.strip(),
                            category="technical_debt",
                            type=debt_type.lower(),
                            files_affected=[file_path],
                            estimated_effort=self._estimate_debt_effort(content),
                            priority=severity,
                            user_business_value=3,
                            time_criticality=self._assess_time_criticality(debt_type),
                            risk_reduction=5,
                            opportunity_enablement=4,
                            job_size=self._story_points_from_effort(2),
                            impact=4,
                            confidence=8,
                            ease=6,
                            debt_impact=self._calculate_debt_impact(content),
                            debt_interest=self._calculate_debt_interest(content),
                            hotspot_multiplier=self._get_file_hotspot_multiplier(file_path),
                            security_severity="none",
                            compliance_impact=False,
                            breaking_change_risk=0.1
                        ))
        except Exception as e:
            print(f"Git history analysis failed: {e}")
        
        return items
    
    def _discover_from_static_analysis(self) -> List[WorkItem]:
        """Run static analysis tools and extract improvement opportunities"""
        items = []
        
        # MyPy strict type checking
        items.extend(self._run_mypy_analysis())
        
        # Security analysis with Bandit
        items.extend(self._run_bandit_analysis())
        
        # Code quality with Ruff
        items.extend(self._run_ruff_analysis())
        
        return items
    
    def _run_mypy_analysis(self) -> List[WorkItem]:
        """Analyze type coverage and strict typing opportunities"""
        items = []
        
        try:
            result = subprocess.run([
                'mypy', 'src/', '--strict', '--show-error-codes'
            ], capture_output=True, text=True)
            
            if result.returncode != 0:
                error_lines = result.stdout.split('\n')
                type_errors = [line for line in error_lines if 'error:' in line]
                
                if len(type_errors) > 0:
                    items.append(WorkItem(
                        id="mypy-001",
                        title=f"Fix {len(type_errors)} MyPy strict type checking errors",
                        description=f"Improve type safety by addressing {len(type_errors)} type checking issues",
                        category="technical_debt",
                        type="type_safety",
                        files_affected=self._extract_files_from_mypy_errors(error_lines),
                        estimated_effort=len(type_errors) * 0.5,
                        priority="medium",
                        user_business_value=4,
                        time_criticality=3,
                        risk_reduction=7,
                        opportunity_enablement=6,
                        job_size=len(type_errors) * 0.5,
                        impact=6,
                        confidence=9,
                        ease=7,
                        debt_impact=len(type_errors) * 0.25,
                        debt_interest=len(type_errors) * 0.1,
                        hotspot_multiplier=1.5,
                        security_severity="none",
                        compliance_impact=False,
                        breaking_change_risk=0.2
                    ))
        except Exception as e:
            print(f"MyPy analysis failed: {e}")
        
        return items
    
    def _run_bandit_analysis(self) -> List[WorkItem]:
        """Security vulnerability analysis"""
        items = []
        
        try:
            result = subprocess.run([
                'bandit', '-r', 'src/', '-f', 'json'
            ], capture_output=True, text=True)
            
            if result.stdout:
                data = json.loads(result.stdout)
                results = data.get('results', [])
                
                high_severity = [r for r in results if r.get('issue_severity') == 'HIGH']
                medium_severity = [r for r in results if r.get('issue_severity') == 'MEDIUM']
                
                if high_severity:
                    items.append(WorkItem(
                        id="security-001",
                        title=f"Fix {len(high_severity)} high-severity security issues",
                        description="Address critical security vulnerabilities identified by Bandit",
                        category="security",
                        type="vulnerability_fix",
                        files_affected=[r['filename'] for r in high_severity],
                        estimated_effort=len(high_severity) * 2,
                        priority="critical",
                        user_business_value=10,
                        time_criticality=9,
                        risk_reduction=10,
                        opportunity_enablement=5,
                        job_size=len(high_severity) * 2,
                        impact=9,
                        confidence=9,
                        ease=6,
                        debt_impact=0,
                        debt_interest=len(high_severity) * 2,
                        hotspot_multiplier=2.0,
                        security_severity="critical",
                        compliance_impact=True,
                        breaking_change_risk=0.3
                    ))
                
                if medium_severity:
                    items.append(WorkItem(
                        id="security-002", 
                        title=f"Fix {len(medium_severity)} medium-severity security issues",
                        description="Address moderate security vulnerabilities identified by Bandit",
                        category="security",
                        type="vulnerability_fix",
                        files_affected=[r['filename'] for r in medium_severity],
                        estimated_effort=len(medium_severity) * 1,
                        priority="high",
                        user_business_value=8,
                        time_criticality=6,
                        risk_reduction=8,
                        opportunity_enablement=4,
                        job_size=len(medium_severity) * 1,
                        impact=7,
                        confidence=8,
                        ease=7,
                        debt_impact=0,
                        debt_interest=len(medium_severity) * 1,
                        hotspot_multiplier=1.5,
                        security_severity="medium",
                        compliance_impact=True,
                        breaking_change_risk=0.2
                    ))
        except Exception as e:
            print(f"Bandit analysis failed: {e}")
        
        return items
    
    def _run_ruff_analysis(self) -> List[WorkItem]:
        """Code quality and style analysis"""
        items = []
        
        try:
            result = subprocess.run([
                'ruff', 'check', 'src/', '--output-format=json'
            ], capture_output=True, text=True)
            
            if result.stdout:
                violations = json.loads(result.stdout)
                
                if len(violations) > 10:  # Only worth addressing if significant
                    items.append(WorkItem(
                        id="quality-001",
                        title=f"Fix {len(violations)} code quality violations",
                        description="Improve code quality by addressing Ruff violations",
                        category="technical_debt",
                        type="code_quality",
                        files_affected=list(set([v['filename'] for v in violations])),
                        estimated_effort=len(violations) * 0.1,
                        priority="medium",
                        user_business_value=3,
                        time_criticality=2,
                        risk_reduction=4,
                        opportunity_enablement=5,
                        job_size=len(violations) * 0.1,
                        impact=4,
                        confidence=9,
                        ease=8,
                        debt_impact=len(violations) * 0.05,
                        debt_interest=len(violations) * 0.02,
                        hotspot_multiplier=1.2,
                        security_severity="none", 
                        compliance_impact=False,
                        breaking_change_risk=0.05
                    ))
                    
        except Exception as e:
            print(f"Ruff analysis failed: {e}")
        
        return items
    
    def _discover_from_security_scans(self) -> List[WorkItem]:
        """Security vulnerability scanning"""
        items = []
        
        # Safety check for known vulnerabilities
        try:
            result = subprocess.run(['safety', 'check', '--json'], 
                                  capture_output=True, text=True)
            if result.stdout:
                data = json.loads(result.stdout)
                vulnerabilities = data.get('vulnerabilities', [])
                
                if vulnerabilities:
                    critical_vulns = [v for v in vulnerabilities if 'critical' in v.get('severity', '').lower()]
                    high_vulns = [v for v in vulnerabilities if 'high' in v.get('severity', '').lower()]
                    
                    if critical_vulns or high_vulns:
                        items.append(WorkItem(
                            id="vuln-001",
                            title=f"Update {len(critical_vulns + high_vulns)} vulnerable dependencies",
                            description="Address critical and high severity dependency vulnerabilities",
                            category="security",
                            type="dependency_vulnerability",
                            files_affected=["requirements.txt", "pyproject.toml"],
                            estimated_effort=len(critical_vulns + high_vulns) * 0.5,
                            priority="critical" if critical_vulns else "high",
                            user_business_value=9,
                            time_criticality=8,
                            risk_reduction=9,
                            opportunity_enablement=3,
                            job_size=len(critical_vulns + high_vulns) * 0.5,
                            impact=8,
                            confidence=9,
                            ease=7,
                            debt_impact=0,
                            debt_interest=len(critical_vulns + high_vulns) * 1.5,
                            hotspot_multiplier=1.0,
                            security_severity="critical" if critical_vulns else "high",
                            compliance_impact=True,
                            breaking_change_risk=0.4
                        ))
        except Exception as e:
            print(f"Safety vulnerability scan failed: {e}")
        
        return items
    
    def _discover_from_performance_data(self) -> List[WorkItem]:
        """Analyze performance metrics for optimization opportunities"""
        items = []
        
        # Check if performance baseline exists
        perf_file = Path("performance-regression.yml")
        if perf_file.exists():
            # Simulate performance analysis
            items.append(WorkItem(
                id="perf-001",
                title="Establish performance baseline measurements",
                description="Run comprehensive performance benchmarks to establish baseline metrics",
                category="performance",
                type="baseline_establishment",
                files_affected=["tests/performance/"],
                estimated_effort=4,
                priority="medium",
                user_business_value=6,
                time_criticality=4,
                risk_reduction=5,
                opportunity_enablement=8,
                job_size=4,
                impact=7,
                confidence=8,
                ease=6,
                debt_impact=0,
                debt_interest=0,
                hotspot_multiplier=1.0,
                security_severity="none",
                compliance_impact=False,
                breaking_change_risk=0.1
            ))
        
        return items
    
    def _discover_from_dependencies(self) -> List[WorkItem]:
        """Analyze dependency freshness and update opportunities"""
        items = []
        
        try:
            # Check for outdated packages
            result = subprocess.run(['pip', 'list', '--outdated', '--format=json'],
                                  capture_output=True, text=True)
            if result.stdout:
                outdated = json.loads(result.stdout)
                
                if len(outdated) > 5:  # Only worth addressing if many outdated
                    items.append(WorkItem(
                        id="deps-001",
                        title=f"Update {len(outdated)} outdated dependencies",
                        description="Update outdated packages to latest stable versions",
                        category="infrastructure",
                        type="dependency_update",
                        files_affected=["requirements.txt", "pyproject.toml"],
                        estimated_effort=len(outdated) * 0.2,
                        priority="medium",
                        user_business_value=4,
                        time_criticality=3,
                        risk_reduction=6,
                        opportunity_enablement=7,
                        job_size=len(outdated) * 0.2,
                        impact=5,
                        confidence=7,
                        ease=6,
                        debt_impact=len(outdated) * 0.1,
                        debt_interest=len(outdated) * 0.05,
                        hotspot_multiplier=1.0,
                        security_severity="none",
                        compliance_impact=False,
                        breaking_change_risk=0.3
                    ))
        except Exception as e:
            print(f"Dependency analysis failed: {e}")
        
        return items
    
    def _discover_from_configurations(self) -> List[WorkItem]:
        """Analyze configuration completeness and optimization opportunities"""
        items = []
        
        # Check for CI/CD pipeline activation (critical gap identified)
        github_workflows = Path(".github/workflows")
        docs_workflows = Path("docs/workflows")
        
        if not github_workflows.exists() and docs_workflows.exists():
            items.append(WorkItem(
                id="cicd-001",
                title="Activate GitHub Actions CI/CD pipeline",
                description="Move workflow templates from docs/workflows to .github/workflows and configure",
                category="infrastructure",
                type="ci_cd_activation",
                files_affected=["docs/workflows/", ".github/workflows/"],
                estimated_effort=8,
                priority="critical",
                user_business_value=10,
                time_criticality=9,
                risk_reduction=8,
                opportunity_enablement=10,
                job_size=8,
                impact=10,
                confidence=9,
                ease=7,
                debt_impact=0,
                debt_interest=0,
                hotspot_multiplier=1.0,
                security_severity="none",
                compliance_impact=True,
                breaking_change_risk=0.2
            ))
        
        # Check for monitoring deployment
        if Path("monitoring/prometheus.yml").exists():
            items.append(WorkItem(
                id="monitor-001", 
                title="Deploy live monitoring infrastructure",
                description="Set up live Prometheus/Grafana deployment for real-time monitoring",
                category="infrastructure",
                type="monitoring_deployment",
                files_affected=["monitoring/", "docker-compose.yml"],
                estimated_effort=6,
                priority="high",
                user_business_value=8,
                time_criticality=6,
                risk_reduction=7,
                opportunity_enablement=8,
                job_size=6,
                impact=8,
                confidence=8,
                ease=6,
                debt_impact=0,
                debt_interest=0,
                hotspot_multiplier=1.0,
                security_severity="none",
                compliance_impact=False,
                breaking_change_risk=0.1
            ))
        
        return items
    
    def calculate_scores(self, item: WorkItem) -> WorkItem:
        """Calculate WSJF, ICE, Technical Debt, and Composite scores"""
        
        # WSJF Score
        cost_of_delay = (
            item.user_business_value + 
            item.time_criticality + 
            item.risk_reduction + 
            item.opportunity_enablement
        )
        item.wsjf_score = cost_of_delay / max(item.job_size, 0.1)
        
        # ICE Score  
        item.ice_score = item.impact * item.confidence * item.ease
        
        # Technical Debt Score
        item.technical_debt_score = (
            (item.debt_impact + item.debt_interest) * item.hotspot_multiplier
        )
        
        # Get adaptive weights for advanced repository
        weights = self.config['scoring']['weights']['advanced']
        
        # Normalize scores for composite calculation
        normalized_wsjf = min(item.wsjf_score / 100, 1.0)
        normalized_ice = min(item.ice_score / 1000, 1.0)  
        normalized_debt = min(item.technical_debt_score / 100, 1.0)
        
        # Calculate composite score
        item.composite_score = (
            weights['wsjf'] * normalized_wsjf +
            weights['ice'] * normalized_ice +
            weights['technicalDebt'] * normalized_debt +
            weights['security'] * 0.1
        ) * 100
        
        # Apply boosts
        if item.security_severity in ['critical', 'high']:
            item.composite_score *= self.config['scoring']['thresholds']['securityBoost']
        
        if item.compliance_impact:
            item.composite_score *= self.config['scoring']['thresholds']['complianceBoost']
            
        if item.category == 'performance':
            item.composite_score *= self.config['scoring']['thresholds']['performanceBoost']
        
        return item
    
    def select_next_best_value(self, scored_items: List[WorkItem]) -> WorkItem:
        """Select the highest-value item for execution"""
        
        # Filter by minimum score threshold
        min_score = self.config['scoring']['thresholds']['minScore']
        qualified_items = [item for item in scored_items if item.composite_score >= min_score]
        
        # Sort by composite score descending
        qualified_items.sort(key=lambda x: x.composite_score, reverse=True)
        
        # Apply additional filters
        for item in qualified_items:
            # Skip if risk too high
            if item.breaking_change_risk > self.config['scoring']['thresholds']['maxRisk']:
                continue
                
            # This is our next best value item
            return item
        
        # No items qualify - return None or generate housekeeping task
        return None
    
    # Helper methods
    def _extract_debt_type(self, content: str) -> str:
        content_lower = content.lower()
        if 'todo' in content_lower:
            return 'TODO'
        elif 'fixme' in content_lower:
            return 'FIXME'
        elif 'hack' in content_lower:
            return 'HACK'
        elif 'xxx' in content_lower:
            return 'XXX'
        elif 'deprecated' in content_lower:
            return 'DEPRECATED'
        else:
            return 'TEMP'
    
    def _assess_debt_severity(self, content: str, file_path: str) -> str:
        content_lower = content.lower()
        if any(word in content_lower for word in ['critical', 'urgent', 'security', 'bug']):
            return 'high'
        elif any(word in content_lower for word in ['important', 'refactor', 'cleanup']):
            return 'medium'
        else:
            return 'low'
    
    def _estimate_debt_effort(self, content: str) -> float:
        """Estimate effort in hours based on debt comment complexity"""
        if len(content) > 100:
            return 4.0
        elif len(content) > 50:
            return 2.0
        else:
            return 1.0
    
    def _assess_time_criticality(self, debt_type: str) -> int:
        criticality_map = {
            'FIXME': 7,
            'HACK': 6,
            'XXX': 5,
            'TODO': 3,
            'DEPRECATED': 4,
            'TEMP': 5
        }
        return criticality_map.get(debt_type, 3)
    
    def _story_points_from_effort(self, hours: float) -> float:
        """Convert effort hours to story points (assuming 6 hours = 1 SP)"""
        return hours / 6.0
    
    def _calculate_debt_impact(self, content: str) -> float:
        """Calculate maintenance time saved by addressing this debt"""
        return len(content) / 20  # Simple heuristic
    
    def _calculate_debt_interest(self, content: str) -> float:
        """Calculate future cost growth if debt is not addressed"""
        return len(content) / 50
    
    def _get_file_hotspot_multiplier(self, file_path: str) -> float:
        """Get hotspot multiplier based on file change frequency"""
        # Simplified - would use git log analysis in real implementation
        if 'core' in file_path or 'main' in file_path:
            return 2.0
        elif 'test' in file_path:
            return 1.0
        else:
            return 1.5
    
    def _extract_files_from_mypy_errors(self, error_lines: List[str]) -> List[str]:
        """Extract file paths from MyPy error output"""
        files = set()
        for line in error_lines:
            if 'error:' in line:
                parts = line.split(':')
                if len(parts) > 0 and parts[0].endswith('.py'):
                    files.add(parts[0])
        return list(files)

def main():
    """Main execution function for value discovery"""
    engine = ValueDiscoveryEngine()
    
    print("🔍 Starting Terragon Value Discovery...")
    
    # Discover work items
    items = engine.discover_work_items()
    print(f"📊 Discovered {len(items)} potential work items")
    
    # Calculate scores for all items
    scored_items = [engine.calculate_scores(item) for item in items]
    
    # Select next best value item
    next_item = engine.select_next_best_value(scored_items)
    
    if next_item:
        print(f"\n🎯 Next Best Value Item:")
        print(f"   Title: {next_item.title}")
        print(f"   Composite Score: {next_item.composite_score:.2f}")
        print(f"   Priority: {next_item.priority}")
        print(f"   Estimated Effort: {next_item.estimated_effort} hours")
        print(f"   Category: {next_item.category}")
    else:
        print("\n✅ No high-value items identified - repository is well optimized!")
    
    # Update metrics
    engine.metrics['current_backlog']['total_items'] = len(scored_items)
    engine.metrics['next_value_discovery']['scheduled_scan'] = (
        datetime.now() + timedelta(hours=1)
    ).isoformat()
    engine._save_metrics()
    
    return scored_items, next_item

if __name__ == "__main__":
    main()