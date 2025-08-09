#!/usr/bin/env python3
"""Security assessment script for Agent Mesh SDLC implementation."""

import os
import re
import ast
from pathlib import Path

class SecurityAssessment:
    """Comprehensive security assessment."""
    
    def __init__(self):
        self.issues = []
        self.passed_checks = []
        self.src_path = Path(__file__).parent.parent / "src"
    
    def assess_security(self):
        """Run comprehensive security assessment."""
        print("🔒 AGENT MESH SECURITY ASSESSMENT")
        print("=" * 40)
        
        # Check for security best practices
        self._check_input_validation()
        self._check_authentication_patterns()
        self._check_cryptographic_practices()
        self._check_secrets_management()
        self._check_network_security()
        self._check_error_handling_security()
        
        # Generate report
        self._generate_security_report()
    
    def _check_input_validation(self):
        """Check for input validation patterns."""
        print("\n🧪 Checking input validation...")
        
        validation_patterns = [
            r'pydantic.*BaseModel',
            r'def.*validate.*\(',
            r'ValidationError',
            r'assert.*isinstance'
        ]
        
        validation_files = 0
        for py_file in self.src_path.rglob("*.py"):
            with open(py_file, 'r') as f:
                content = f.read()
                for pattern in validation_patterns:
                    if re.search(pattern, content, re.IGNORECASE):
                        validation_files += 1
                        break
        
        if validation_files >= 3:
            self.passed_checks.append("Input validation patterns implemented")
            print("   ✅ Input validation patterns found")
        else:
            self.issues.append("Limited input validation patterns")
            print("   ⚠️  Limited input validation patterns")
    
    def _check_authentication_patterns(self):
        """Check for authentication and authorization patterns."""
        print("\n🧪 Checking authentication patterns...")
        
        auth_patterns = [
            r'authenticate',
            r'authorize',
            r'token.*verify',
            r'permission',
            r'access.*control'
        ]
        
        auth_implementations = 0
        for py_file in self.src_path.rglob("*.py"):
            with open(py_file, 'r') as f:
                content = f.read()
                for pattern in auth_patterns:
                    if re.search(pattern, content, re.IGNORECASE):
                        auth_implementations += 1
                        break
        
        if auth_implementations >= 2:
            self.passed_checks.append("Authentication patterns implemented")
            print("   ✅ Authentication patterns found")
        else:
            self.issues.append("Authentication patterns need enhancement")
            print("   ⚠️  Authentication patterns need enhancement")
    
    def _check_cryptographic_practices(self):
        """Check for proper cryptographic practices."""
        print("\n🧪 Checking cryptographic practices...")
        
        crypto_patterns = [
            r'cryptography',
            r'hashlib',
            r'secrets',
            r'nacl',
            r'signature',
            r'encrypt'
        ]
        
        crypto_usage = 0
        weak_crypto = 0
        
        for py_file in self.src_path.rglob("*.py"):
            with open(py_file, 'r') as f:
                content = f.read()
                
                # Check for good crypto
                for pattern in crypto_patterns:
                    if re.search(pattern, content, re.IGNORECASE):
                        crypto_usage += 1
                        break
                
                # Check for weak crypto
                weak_patterns = [r'md5\(', r'sha1\(']
                for pattern in weak_patterns:
                    if re.search(pattern, content):
                        weak_crypto += 1
        
        if crypto_usage > 0 and weak_crypto == 0:
            self.passed_checks.append("Strong cryptographic practices")
            print("   ✅ Strong cryptographic practices")
        elif weak_crypto > 0:
            self.issues.append("Weak cryptographic algorithms detected")
            print("   ❌ Weak cryptographic algorithms detected")
        else:
            self.issues.append("Limited cryptographic implementation")
            print("   ⚠️  Limited cryptographic implementation")
    
    def _check_secrets_management(self):
        """Check for secrets management."""
        print("\n🧪 Checking secrets management...")
        
        # Check for hardcoded secrets
        hardcoded_patterns = [
            r'password\s*=\s*["\'][^"\']{8,}["\']',
            r'secret\s*=\s*["\'][^"\']{8,}["\']',
            r'api[_-]?key\s*=\s*["\'][^"\']{8,}["\']',
            r'token\s*=\s*["\'][^"\']{20,}["\']'
        ]
        
        hardcoded_secrets = 0
        env_usage = 0
        
        for py_file in self.src_path.rglob("*.py"):
            with open(py_file, 'r') as f:
                content = f.read()
                
                # Check for hardcoded secrets
                for pattern in hardcoded_patterns:
                    if re.search(pattern, content, re.IGNORECASE):
                        hardcoded_secrets += 1
                
                # Check for environment variable usage
                if re.search(r'os\.environ|getenv', content):
                    env_usage += 1
        
        if hardcoded_secrets == 0:
            self.passed_checks.append("No hardcoded secrets found")
            print("   ✅ No hardcoded secrets found")
        else:
            self.issues.append(f"Found {hardcoded_secrets} potential hardcoded secrets")
            print(f"   ❌ Found {hardcoded_secrets} potential hardcoded secrets")
        
        if env_usage > 0:
            self.passed_checks.append("Environment variable usage for configuration")
            print("   ✅ Environment variable usage found")
    
    def _check_network_security(self):
        """Check for network security practices."""
        print("\n🧪 Checking network security...")
        
        security_features = []
        
        for py_file in self.src_path.rglob("*.py"):
            with open(py_file, 'r') as f:
                content = f.read()
                
                if re.search(r'ssl|tls', content, re.IGNORECASE):
                    security_features.append("SSL/TLS")
                
                if re.search(r'rate.*limit', content, re.IGNORECASE):
                    security_features.append("Rate Limiting")
                
                if re.search(r'timeout', content, re.IGNORECASE):
                    security_features.append("Timeout Controls")
        
        if len(security_features) >= 2:
            self.passed_checks.append(f"Network security features: {', '.join(security_features)}")
            print(f"   ✅ Network security features: {', '.join(security_features)}")
        else:
            self.issues.append("Limited network security features")
            print("   ⚠️  Limited network security features")
    
    def _check_error_handling_security(self):
        """Check for secure error handling."""
        print("\n🧪 Checking secure error handling...")
        
        secure_patterns = 0
        info_leakage = 0
        
        for py_file in self.src_path.rglob("*.py"):
            with open(py_file, 'r') as f:
                content = f.read()
                
                # Check for secure logging
                if re.search(r'logger.*(?:debug|info)', content):
                    secure_patterns += 1
                
                # Check for potential information leakage
                leakage_patterns = [
                    r'print\s*\(\s*.*password',
                    r'print\s*\(\s*.*secret',
                    r'logger.*error.*(?:password|secret|token)'
                ]
                
                for pattern in leakage_patterns:
                    if re.search(pattern, content, re.IGNORECASE):
                        info_leakage += 1
        
        if info_leakage == 0:
            self.passed_checks.append("No obvious information leakage in error handling")
            print("   ✅ No obvious information leakage detected")
        else:
            self.issues.append(f"Potential information leakage: {info_leakage} instances")
            print(f"   ❌ Potential information leakage: {info_leakage} instances")
    
    def _generate_security_report(self):
        """Generate final security assessment report."""
        print("\n" + "=" * 50)
        print("🔒 SECURITY ASSESSMENT REPORT")
        print("=" * 50)
        
        total_checks = len(self.passed_checks) + len(self.issues)
        security_score = (len(self.passed_checks) / total_checks * 100) if total_checks > 0 else 0
        
        print(f"\n📊 SECURITY SCORE: {security_score:.1f}%")
        print(f"   Passed Checks: {len(self.passed_checks)}")
        print(f"   Security Issues: {len(self.issues)}")
        
        if self.passed_checks:
            print(f"\n✅ PASSED SECURITY CHECKS ({len(self.passed_checks)}):")
            for check in self.passed_checks:
                print(f"   • {check}")
        
        if self.issues:
            print(f"\n⚠️  SECURITY ISSUES ({len(self.issues)}):")
            for issue in self.issues:
                print(f"   • {issue}")
        
        print(f"\n🎯 SECURITY ASSESSMENT:")
        if security_score >= 80:
            print("   🟢 STRONG SECURITY POSTURE")
            print("   System demonstrates good security practices")
        elif security_score >= 60:
            print("   🟡 MODERATE SECURITY POSTURE")
            print("   Some security improvements recommended")
        else:
            print("   🔴 SECURITY NEEDS IMPROVEMENT")
            print("   Significant security enhancements required")
        
        return security_score


def main():
    """Run security assessment."""
    assessment = SecurityAssessment()
    score = assessment.assess_security()
    
    # Return appropriate exit code
    return 0 if score and score >= 60 else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())