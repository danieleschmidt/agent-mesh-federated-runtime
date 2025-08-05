"""Advanced input validation and sanitization for Agent Mesh.

This module provides comprehensive validation, sanitization, and security
checks for all inputs and data flowing through the system.
"""

import re
import ipaddress
import hashlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Union, Type, Callable
from uuid import UUID
from enum import Enum

import structlog
from pydantic import BaseModel, ValidationError, validator, Field

from .error_handling import ValidationError as AgentValidationError


class ValidationLevel(Enum):
    """Validation strictness levels."""
    
    PERMISSIVE = "permissive"
    STANDARD = "standard"
    STRICT = "strict"
    PARANOID = "paranoid"


class SanitizationAction(Enum):
    """Actions to take during sanitization."""
    
    ALLOW = "allow"
    SANITIZE = "sanitize"
    REJECT = "reject"
    LOG_AND_ALLOW = "log_and_allow"


@dataclass
class ValidationRule:
    """Validation rule definition."""
    
    field_name: str
    rule_type: str
    parameters: Dict[str, Any]
    action: SanitizationAction = SanitizationAction.REJECT
    message: str = "Validation failed"


@dataclass
class ValidationResult:
    """Result of validation operation."""
    
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    sanitized_data: Optional[Any] = None
    metadata: Dict[str, Any] = None


class SecurityPatterns:
    """Security patterns for detection and prevention."""
    
    # SQL injection patterns
    SQL_INJECTION_PATTERNS = [
        r'(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|UNION)\b)',
        r'(--|;|\/\*|\*\/)',
        r'(\b(OR|AND)\s+\d+\s*=\s*\d+)',
        r'(\b(OR|AND)\s+[\'\"]\w+[\'\"]\s*=\s*[\'\"]\w+[\'\"])'
    ]
    
    # XSS patterns
    XSS_PATTERNS = [
        r'<\s*script[^>]*>.*?</\s*script\s*>',
        r'<\s*iframe[^>]*>.*?</\s*iframe\s*>',
        r'javascript\s*:',
        r'on\w+\s*=',
        r'<\s*object[^>]*>.*?</\s*object\s*>'
    ]
    
    # Command injection patterns
    COMMAND_INJECTION_PATTERNS = [
        r'[;&|`$(){}\\\\]',
        r'\b(cat|ls|pwd|whoami|id|uname|wget|curl|nc|netcat)\b',
        r'(>|<|>>|<<)',
        r'\$\{.*\}',
        r'`.*`'
    ]
    
    # Path traversal patterns
    PATH_TRAVERSAL_PATTERNS = [
        r'\.\./|\.\.\\',
        r'/etc/passwd',
        r'/proc/',
        r'\\windows\\system32',
        r'%2e%2e%2f',
        r'%2e%2e%5c'
    ]
    
    # File extension blacklist
    DANGEROUS_EXTENSIONS = {
        '.exe', '.bat', '.cmd', '.com', '.pif', '.scr', '.vbs', '.js',
        '.jar', '.app', '.deb', '.pkg', '.dmg', '.iso', '.sh', '.ps1'
    }


class InputSanitizer:
    """Advanced input sanitizer with multiple strategies."""
    
    def __init__(self, validation_level: ValidationLevel = ValidationLevel.STANDARD):
        self.validation_level = validation_level
        self.logger = structlog.get_logger("input_sanitizer")
        
        # Compile security patterns
        self.sql_patterns = [re.compile(p, re.IGNORECASE) for p in SecurityPatterns.SQL_INJECTION_PATTERNS]
        self.xss_patterns = [re.compile(p, re.IGNORECASE) for p in SecurityPatterns.XSS_PATTERNS]
        self.cmd_patterns = [re.compile(p, re.IGNORECASE) for p in SecurityPatterns.COMMAND_INJECTION_PATTERNS]
        self.path_patterns = [re.compile(p, re.IGNORECASE) for p in SecurityPatterns.PATH_TRAVERSAL_PATTERNS]
    
    def sanitize_string(self, value: str, max_length: int = 1000) -> ValidationResult:
        """Sanitize string input with security checks."""
        errors = []
        warnings = []
        sanitized = value
        
        # Length check
        if len(value) > max_length:
            if self.validation_level in [ValidationLevel.STRICT, ValidationLevel.PARANOID]:
                errors.append(f"String too long: {len(value)} > {max_length}")
            else:
                sanitized = value[:max_length]
                warnings.append(f"String truncated to {max_length} characters")
        
        # Security pattern checks
        security_issues = self._check_security_patterns(value)
        if security_issues:
            if self.validation_level == ValidationLevel.PARANOID:
                errors.extend(security_issues)
            else:
                warnings.extend(security_issues)
                sanitized = self._remove_dangerous_patterns(sanitized)
        
        # Character encoding validation
        try:
            sanitized.encode('utf-8')
        except UnicodeEncodeError:
            errors.append("Invalid UTF-8 encoding")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            sanitized_data=sanitized
        )
    
    def sanitize_dict(self, data: Dict[str, Any], schema: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """Sanitize dictionary with nested validation."""
        errors = []
        warnings = []
        sanitized = {}
        
        for key, value in data.items():
            # Validate key
            key_result = self.sanitize_string(key, 100)
            if not key_result.is_valid:
                errors.extend([f"Key '{key}': {error}" for error in key_result.errors])
                continue
            
            warnings.extend([f"Key '{key}': {warning}" for warning in key_result.warnings])
            clean_key = key_result.sanitized_data
            
            # Validate value based on type
            if isinstance(value, str):
                value_result = self.sanitize_string(value)
            elif isinstance(value, dict):
                value_result = self.sanitize_dict(value)
            elif isinstance(value, list):
                value_result = self.sanitize_list(value)
            else:
                value_result = self._sanitize_primitive(value)
            
            if not value_result.is_valid:
                errors.extend([f"Value for '{key}': {error}" for error in value_result.errors])
                continue
            
            warnings.extend([f"Value for '{key}': {warning}" for warning in value_result.warnings])
            sanitized[clean_key] = value_result.sanitized_data
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            sanitized_data=sanitized
        )
    
    def sanitize_list(self, items: List[Any], max_items: int = 1000) -> ValidationResult:
        """Sanitize list with item validation."""
        errors = []
        warnings = []
        sanitized = []
        
        if len(items) > max_items:
            if self.validation_level in [ValidationLevel.STRICT, ValidationLevel.PARANOID]:
                errors.append(f"Too many items: {len(items)} > {max_items}")
                return ValidationResult(is_valid=False, errors=errors, warnings=warnings)
            else:
                items = items[:max_items]
                warnings.append(f"List truncated to {max_items} items")
        
        for i, item in enumerate(items):
            if isinstance(item, str):
                item_result = self.sanitize_string(item)
            elif isinstance(item, dict):
                item_result = self.sanitize_dict(item)
            elif isinstance(item, list):
                item_result = self.sanitize_list(item, max_items // 10)  # Nested lists smaller
            else:
                item_result = self._sanitize_primitive(item)
            
            if not item_result.is_valid:
                errors.extend([f"Item {i}: {error}" for error in item_result.errors])
                continue
            
            warnings.extend([f"Item {i}: {warning}" for warning in item_result.warnings])
            sanitized.append(item_result.sanitized_data)
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            sanitized_data=sanitized
        )
    
    def validate_network_address(self, address: str) -> ValidationResult:
        """Validate network address (IP or hostname)."""
        errors = []
        warnings = []
        
        # Check for private/local addresses in paranoid mode
        try:
            ip = ipaddress.ip_address(address)
            if self.validation_level == ValidationLevel.PARANOID:
                if ip.is_private or ip.is_loopback or ip.is_link_local:
                    warnings.append(f"Private/local IP address: {address}")
        except ValueError:
            # Not an IP, check as hostname
            if not self._is_valid_hostname(address):
                errors.append(f"Invalid hostname: {address}")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            sanitized_data=address
        )
    
    def validate_uuid(self, uuid_str: str) -> ValidationResult:
        """Validate UUID format."""
        try:
            UUID(uuid_str)
            return ValidationResult(is_valid=True, errors=[], warnings=[], sanitized_data=uuid_str)
        except ValueError:
            return ValidationResult(
                is_valid=False,
                errors=[f"Invalid UUID format: {uuid_str}"],
                warnings=[],
                sanitized_data=None
            )
    
    def validate_file_path(self, path: str) -> ValidationResult:
        """Validate file path for security issues."""
        errors = []
        warnings = []
        
        # Path traversal check
        for pattern in self.path_patterns:
            if pattern.search(path):
                errors.append(f"Path traversal detected: {path}")
                break
        
        # Dangerous file extension check
        import os
        _, ext = os.path.splitext(path.lower())
        if ext in SecurityPatterns.DANGEROUS_EXTENSIONS:
            if self.validation_level == ValidationLevel.PARANOID:
                errors.append(f"Dangerous file extension: {ext}")
            else:
                warnings.append(f"Potentially dangerous file extension: {ext}")
        
        # Absolute path check
        if os.path.isabs(path) and self.validation_level == ValidationLevel.PARANOID:
            warnings.append("Absolute path detected")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            sanitized_data=path
        )
    
    # Private methods
    
    def _check_security_patterns(self, value: str) -> List[str]:
        """Check for security patterns in string."""
        issues = []
        
        # SQL injection check
        for pattern in self.sql_patterns:
            if pattern.search(value):
                issues.append("Potential SQL injection detected")
                break
        
        # XSS check
        for pattern in self.xss_patterns:
            if pattern.search(value):
                issues.append("Potential XSS detected")
                break
        
        # Command injection check
        for pattern in self.cmd_patterns:
            if pattern.search(value):
                issues.append("Potential command injection detected")
                break
        
        return issues
    
    def _remove_dangerous_patterns(self, value: str) -> str:
        """Remove dangerous patterns from string."""
        sanitized = value
        
        # Remove script tags
        sanitized = re.sub(r'<\s*script[^>]*>.*?</\s*script\s*>', '', sanitized, flags=re.IGNORECASE | re.DOTALL)
        
        # Remove iframe tags
        sanitized = re.sub(r'<\s*iframe[^>]*>.*?</\s*iframe\s*>', '', sanitized, flags=re.IGNORECASE | re.DOTALL)
        
        # Remove javascript: protocols
        sanitized = re.sub(r'javascript\s*:', '', sanitized, flags=re.IGNORECASE)
        
        # Remove event handlers
        sanitized = re.sub(r'on\w+\s*=\s*["\'][^"\']*["\']', '', sanitized, flags=re.IGNORECASE)
        
        return sanitized
    
    def _sanitize_primitive(self, value: Any) -> ValidationResult:
        """Sanitize primitive values."""
        if isinstance(value, (int, float, bool)):
            # Check for reasonable ranges
            if isinstance(value, (int, float)):
                if abs(value) > 1e15:  # Very large numbers
                    return ValidationResult(
                        is_valid=False,
                        errors=[f"Number too large: {value}"],
                        warnings=[]
                    )
            
            return ValidationResult(
                is_valid=True,
                errors=[],
                warnings=[],
                sanitized_data=value
            )
        
        return ValidationResult(
            is_valid=False,
            errors=[f"Unsupported type: {type(value)}"],
            warnings=[],
            sanitized_data=None
        )
    
    def _is_valid_hostname(self, hostname: str) -> bool:
        """Check if hostname is valid."""
        if len(hostname) > 253:
            return False
        
        if hostname[-1] == ".":
            hostname = hostname[:-1]
        
        allowed = re.compile(r"(?!-)[A-Z\d-]{1,63}(?<!-)$", re.IGNORECASE)
        return all(allowed.match(x) for x in hostname.split("."))


class ModelValidator:
    """Pydantic model-based validator with advanced features."""
    
    def __init__(self, sanitizer: InputSanitizer):
        self.sanitizer = sanitizer
        self.logger = structlog.get_logger("model_validator")
    
    def validate_against_model(self, data: Dict[str, Any], model_class: Type[BaseModel]) -> ValidationResult:
        """Validate data against Pydantic model."""
        try:
            # First sanitize the data
            sanitize_result = self.sanitizer.sanitize_dict(data)
            if not sanitize_result.is_valid:
                return sanitize_result
            
            # Then validate against model
            validated_model = model_class(**sanitize_result.sanitized_data)
            
            return ValidationResult(
                is_valid=True,
                errors=[],
                warnings=sanitize_result.warnings,
                sanitized_data=validated_model.dict(),
                metadata={"model": model_class.__name__}
            )
            
        except ValidationError as e:
            errors = [f"{error['loc'][0] if error['loc'] else 'root'}: {error['msg']}" for error in e.errors()]
            return ValidationResult(
                is_valid=False,
                errors=errors,
                warnings=[],
                sanitized_data=None
            )
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                errors=[f"Validation error: {str(e)}"],
                warnings=[],
                sanitized_data=None
            )


class ComprehensiveValidator:
    """Main validation orchestrator combining all validation strategies."""
    
    def __init__(self, validation_level: ValidationLevel = ValidationLevel.STANDARD):
        self.validation_level = validation_level
        self.sanitizer = InputSanitizer(validation_level)
        self.model_validator = ModelValidator(self.sanitizer)
        self.logger = structlog.get_logger("comprehensive_validator")
        
        # Custom validation rules registry
        self.custom_rules: Dict[str, ValidationRule] = {}
        
        # Validation statistics
        self.validation_stats = {
            "total_validations": 0,
            "successful_validations": 0,
            "failed_validations": 0,
            "sanitizations_applied": 0
        }
    
    def validate(
        self, 
        data: Any, 
        data_type: str = "generic",
        model_class: Optional[Type[BaseModel]] = None,
        custom_rules: Optional[List[ValidationRule]] = None
    ) -> ValidationResult:
        """Comprehensive validation with multiple strategies."""
        self.validation_stats["total_validations"] += 1
        
        try:
            # Apply custom rules first
            if custom_rules:
                rule_result = self._apply_custom_rules(data, custom_rules)
                if not rule_result.is_valid:
                    self.validation_stats["failed_validations"] += 1
                    return rule_result
                data = rule_result.sanitized_data
            
            # Type-specific validation
            if data_type == "string" and isinstance(data, str):
                result = self.sanitizer.sanitize_string(data)
            elif data_type == "dict" and isinstance(data, dict):
                result = self.sanitizer.sanitize_dict(data)
            elif data_type == "list" and isinstance(data, list):
                result = self.sanitizer.sanitize_list(data)
            elif data_type == "network_address" and isinstance(data, str):
                result = self.sanitizer.validate_network_address(data)
            elif data_type == "uuid" and isinstance(data, str):
                result = self.sanitizer.validate_uuid(data)
            elif data_type == "file_path" and isinstance(data, str):
                result = self.sanitizer.validate_file_path(data)
            elif model_class and isinstance(data, dict):
                result = self.model_validator.validate_against_model(data, model_class)
            else:
                # Generic validation
                if isinstance(data, str):
                    result = self.sanitizer.sanitize_string(data)
                elif isinstance(data, dict):
                    result = self.sanitizer.sanitize_dict(data)
                elif isinstance(data, list):
                    result = self.sanitizer.sanitize_list(data)
                else:
                    result = self.sanitizer._sanitize_primitive(data)
            
            # Update statistics
            if result.is_valid:
                self.validation_stats["successful_validations"] += 1
            else:
                self.validation_stats["failed_validations"] += 1
            
            if result.sanitized_data != data:
                self.validation_stats["sanitizations_applied"] += 1
            
            # Log validation events
            if not result.is_valid:
                self.logger.warning("Validation failed",
                                  data_type=data_type,
                                  errors=result.errors)
            elif result.warnings:
                self.logger.info("Validation warnings",
                                data_type=data_type,
                                warnings=result.warnings)
            
            return result
            
        except Exception as e:
            self.validation_stats["failed_validations"] += 1
            self.logger.error("Validation exception", error=str(e))
            
            return ValidationResult(
                is_valid=False,
                errors=[f"Validation exception: {str(e)}"],
                warnings=[],
                sanitized_data=None
            )
    
    def register_custom_rule(self, rule: ValidationRule) -> None:
        """Register custom validation rule."""
        self.custom_rules[rule.field_name] = rule
        self.logger.info("Custom validation rule registered", field=rule.field_name)
    
    def get_validation_statistics(self) -> Dict[str, Any]:
        """Get validation statistics."""
        total = self.validation_stats["total_validations"]
        if total == 0:
            return self.validation_stats
        
        return {
            **self.validation_stats,
            "success_rate": self.validation_stats["successful_validations"] / total,
            "failure_rate": self.validation_stats["failed_validations"] / total,
            "sanitization_rate": self.validation_stats["sanitizations_applied"] / total
        }
    
    # Private methods
    
    def _apply_custom_rules(self, data: Any, rules: List[ValidationRule]) -> ValidationResult:
        """Apply custom validation rules."""
        errors = []
        warnings = []
        sanitized = data
        
        for rule in rules:
            try:
                # Apply rule based on type
                if rule.rule_type == "regex":
                    pattern = rule.parameters.get("pattern")
                    if pattern and isinstance(data, str):
                        if not re.match(pattern, data):
                            if rule.action == SanitizationAction.REJECT:
                                errors.append(rule.message)
                            elif rule.action == SanitizationAction.LOG_AND_ALLOW:
                                warnings.append(rule.message)
                
                elif rule.rule_type == "range":
                    min_val = rule.parameters.get("min")
                    max_val = rule.parameters.get("max")
                    if isinstance(data, (int, float)):
                        if min_val is not None and data < min_val:
                            errors.append(f"{rule.message}: value {data} < {min_val}")
                        if max_val is not None and data > max_val:
                            errors.append(f"{rule.message}: value {data} > {max_val}")
                
                elif rule.rule_type == "length":
                    min_len = rule.parameters.get("min")
                    max_len = rule.parameters.get("max")
                    if hasattr(data, "__len__"):
                        length = len(data)
                        if min_len is not None and length < min_len:
                            errors.append(f"{rule.message}: length {length} < {min_len}")
                        if max_len is not None and length > max_len:
                            if rule.action == SanitizationAction.SANITIZE:
                                sanitized = data[:max_len]
                                warnings.append(f"Data truncated to {max_len} characters")
                            else:
                                errors.append(f"{rule.message}: length {length} > {max_len}")
                
            except Exception as e:
                errors.append(f"Rule application error: {str(e)}")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            sanitized_data=sanitized
        )


# Global validator instance
_global_validator: Optional[ComprehensiveValidator] = None


def get_validator(validation_level: ValidationLevel = ValidationLevel.STANDARD) -> ComprehensiveValidator:
    """Get global validator instance."""
    global _global_validator
    if _global_validator is None:
        _global_validator = ComprehensiveValidator(validation_level)
    return _global_validator


def validate_input(
    data: Any, 
    data_type: str = "generic", 
    validation_level: ValidationLevel = ValidationLevel.STANDARD
) -> ValidationResult:
    """Validate input using global validator."""
    validator = get_validator(validation_level)
    return validator.validate(data, data_type)


def sanitize_for_logging(data: Any) -> str:
    """Sanitize data for safe logging."""
    if isinstance(data, dict):
        # Mask sensitive fields
        sensitive_fields = {"password", "token", "key", "secret", "auth", "credential"}
        sanitized = {}
        for k, v in data.items():
            if any(field in k.lower() for field in sensitive_fields):
                sanitized[k] = "***MASKED***"
            else:
                sanitized[k] = sanitize_for_logging(v) if isinstance(v, (dict, list)) else str(v)[:100]
        return str(sanitized)
    elif isinstance(data, list):
        return str([sanitize_for_logging(item) for item in data[:10]])  # Limit to first 10 items
    else:
        return str(data)[:100]  # Limit string length