"""Cross-platform compatibility and validation system for Agent Mesh.

Provides platform detection, compatibility checking, and environment validation
to ensure consistent behavior across different operating systems and architectures.
"""

import os
import sys
import platform
import subprocess
import asyncio
import shutil
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
import importlib.util

import structlog

logger = structlog.get_logger("platform")


class PlatformType(Enum):
    """Supported platform types."""
    LINUX = "linux"
    DARWIN = "darwin"  # macOS
    WINDOWS = "windows"
    FREEBSD = "freebsd"
    UNKNOWN = "unknown"


class Architecture(Enum):
    """Supported CPU architectures."""
    X86_64 = "x86_64"
    ARM64 = "arm64"
    ARM = "arm"
    I386 = "i386"
    UNKNOWN = "unknown"


class ContainerType(Enum):
    """Container runtime types."""
    DOCKER = "docker"
    PODMAN = "podman"
    CONTAINERD = "containerd"
    KUBERNETES = "kubernetes"
    NONE = "none"


@dataclass
class PlatformInfo:
    """Comprehensive platform information."""
    platform_type: PlatformType
    architecture: Architecture
    python_version: Tuple[int, int, int]
    os_version: str
    kernel_version: str
    container_type: ContainerType
    available_memory: int  # MB
    cpu_count: int
    supports_async: bool
    has_ipv6: bool
    network_interfaces: List[str] = field(default_factory=list)
    environment_vars: Dict[str, str] = field(default_factory=dict)
    path_separator: str = "/"
    line_separator: str = "\n"
    
    def is_unix_like(self) -> bool:
        """Check if platform is Unix-like."""
        return self.platform_type in [PlatformType.LINUX, PlatformType.DARWIN, PlatformType.FREEBSD]
    
    def is_supported(self) -> bool:
        """Check if platform is officially supported."""
        return (self.platform_type != PlatformType.UNKNOWN and 
                self.architecture != Architecture.UNKNOWN and
                self.python_version >= (3, 8))
    
    def get_config_dir(self) -> Path:
        """Get platform-appropriate configuration directory."""
        if self.platform_type == PlatformType.WINDOWS:
            base = Path(os.environ.get('APPDATA', '~'))
        elif self.platform_type == PlatformType.DARWIN:
            base = Path('~/Library/Application Support')
        else:  # Unix-like
            base = Path(os.environ.get('XDG_CONFIG_HOME', '~/.config'))
        
        return (base / 'agent_mesh').expanduser()
    
    def get_data_dir(self) -> Path:
        """Get platform-appropriate data directory."""
        if self.platform_type == PlatformType.WINDOWS:
            base = Path(os.environ.get('LOCALAPPDATA', '~'))
        elif self.platform_type == PlatformType.DARWIN:
            base = Path('~/Library/Application Support')
        else:  # Unix-like
            base = Path(os.environ.get('XDG_DATA_HOME', '~/.local/share'))
        
        return (base / 'agent_mesh').expanduser()
    
    def get_cache_dir(self) -> Path:
        """Get platform-appropriate cache directory."""
        if self.platform_type == PlatformType.WINDOWS:
            base = Path(os.environ.get('TEMP', '~'))
        elif self.platform_type == PlatformType.DARWIN:
            base = Path('~/Library/Caches')
        else:  # Unix-like
            base = Path(os.environ.get('XDG_CACHE_HOME', '~/.cache'))
        
        return (base / 'agent_mesh').expanduser()


class PlatformDetector:
    """Platform detection and compatibility validation."""
    
    def __init__(self):
        self.logger = structlog.get_logger("platform_detector")
        self._platform_info: Optional[PlatformInfo] = None
    
    async def detect_platform(self) -> PlatformInfo:
        """Comprehensive platform detection."""
        if self._platform_info:
            return self._platform_info
        
        self.logger.info("Starting comprehensive platform detection")
        
        # Basic platform detection
        platform_type = self._detect_platform_type()
        architecture = self._detect_architecture()
        python_version = sys.version_info[:3]
        
        # OS information
        os_version = platform.version()
        kernel_version = platform.release()
        
        # Container detection
        container_type = await self._detect_container_type()
        
        # System resources
        available_memory = self._get_available_memory()
        cpu_count = os.cpu_count() or 1
        
        # Capabilities
        supports_async = sys.version_info >= (3, 7)
        has_ipv6 = self._check_ipv6_support()
        
        # Network interfaces
        network_interfaces = await self._get_network_interfaces()
        
        # Environment
        environment_vars = dict(os.environ)
        
        # Platform-specific separators
        path_separator = os.sep
        line_separator = os.linesep
        
        self._platform_info = PlatformInfo(
            platform_type=platform_type,
            architecture=architecture,
            python_version=python_version,
            os_version=os_version,
            kernel_version=kernel_version,
            container_type=container_type,
            available_memory=available_memory,
            cpu_count=cpu_count,
            supports_async=supports_async,
            has_ipv6=has_ipv6,
            network_interfaces=network_interfaces,
            environment_vars=environment_vars,
            path_separator=path_separator,
            line_separator=line_separator
        )
        
        self.logger.info("Platform detection completed",
                        platform=platform_type.value,
                        architecture=architecture.value,
                        python_version=".".join(map(str, python_version)),
                        container=container_type.value,
                        memory_mb=available_memory,
                        cpu_count=cpu_count)
        
        return self._platform_info
    
    def _detect_platform_type(self) -> PlatformType:
        """Detect the operating system platform."""
        system = platform.system().lower()
        
        if system == "linux":
            return PlatformType.LINUX
        elif system == "darwin":
            return PlatformType.DARWIN
        elif system == "windows":
            return PlatformType.WINDOWS
        elif system == "freebsd":
            return PlatformType.FREEBSD
        else:
            return PlatformType.UNKNOWN
    
    def _detect_architecture(self) -> Architecture:
        """Detect CPU architecture."""
        machine = platform.machine().lower()
        
        if machine in ["x86_64", "amd64"]:
            return Architecture.X86_64
        elif machine in ["arm64", "aarch64"]:
            return Architecture.ARM64
        elif machine.startswith("arm"):
            return Architecture.ARM
        elif machine in ["i386", "i686"]:
            return Architecture.I386
        else:
            return Architecture.UNKNOWN
    
    async def _detect_container_type(self) -> ContainerType:
        """Detect if running in a container and what type."""
        try:
            # Check for Kubernetes
            if (Path('/var/run/secrets/kubernetes.io').exists() or
                os.environ.get('KUBERNETES_SERVICE_HOST')):
                return ContainerType.KUBERNETES
            
            # Check for Docker
            if (Path('/.dockerenv').exists() or
                Path('/proc/self/cgroup').exists()):
                try:
                    with open('/proc/self/cgroup', 'r') as f:
                        content = f.read()
                        if 'docker' in content:
                            return ContainerType.DOCKER
                        elif 'containerd' in content:
                            return ContainerType.CONTAINERD
                except:
                    pass
            
            # Check for Podman
            if os.environ.get('container') == 'podman':
                return ContainerType.PODMAN
            
            # Check container runtime commands
            for cmd in ['docker', 'podman', 'containerd']:
                if shutil.which(cmd):
                    try:
                        result = await asyncio.create_subprocess_exec(
                            cmd, '--version',
                            stdout=asyncio.subprocess.PIPE,
                            stderr=asyncio.subprocess.PIPE
                        )
                        await result.communicate()
                        if result.returncode == 0:
                            return ContainerType(cmd)
                    except:
                        continue
            
            return ContainerType.NONE
            
        except Exception as e:
            self.logger.warning("Container detection failed", error=str(e))
            return ContainerType.NONE
    
    def _get_available_memory(self) -> int:
        """Get available system memory in MB."""
        try:
            if hasattr(os, 'sysconf') and hasattr(os, 'sysconf_names'):
                if 'SC_PAGE_SIZE' in os.sysconf_names and 'SC_PHYS_PAGES' in os.sysconf_names:
                    page_size = os.sysconf('SC_PAGE_SIZE')
                    page_count = os.sysconf('SC_PHYS_PAGES')
                    return (page_size * page_count) // (1024 * 1024)
            
            # Alternative methods for different platforms
            if sys.platform == 'win32':
                import ctypes
                kernel32 = ctypes.windll.kernel32
                c_ulong = ctypes.c_ulong
                
                class MEMORYSTATUS(ctypes.Structure):
                    _fields_ = [
                        ('dwLength', c_ulong),
                        ('dwMemoryLoad', c_ulong),
                        ('dwTotalPhys', c_ulong),
                        ('dwAvailPhys', c_ulong),
                        ('dwTotalPageFile', c_ulong),
                        ('dwAvailPageFile', c_ulong),
                        ('dwTotalVirtual', c_ulong),
                        ('dwAvailVirtual', c_ulong),
                    ]
                
                memory_status = MEMORYSTATUS()
                memory_status.dwLength = ctypes.sizeof(MEMORYSTATUS)
                kernel32.GlobalMemoryStatus(ctypes.byref(memory_status))
                return memory_status.dwTotalPhys // (1024 * 1024)
            
            # Fallback: try to read from /proc/meminfo
            if Path('/proc/meminfo').exists():
                with open('/proc/meminfo', 'r') as f:
                    for line in f:
                        if line.startswith('MemTotal:'):
                            return int(line.split()[1]) // 1024
            
            # Default fallback
            return 1024  # 1GB default
            
        except Exception as e:
            self.logger.warning("Memory detection failed", error=str(e))
            return 1024
    
    def _check_ipv6_support(self) -> bool:
        """Check if IPv6 is supported and available."""
        try:
            import socket
            # Try to create an IPv6 socket
            sock = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
            sock.close()
            return True
        except:
            return False
    
    async def _get_network_interfaces(self) -> List[str]:
        """Get available network interfaces."""
        interfaces = []
        
        try:
            if sys.platform != 'win32':
                # Unix-like systems
                result = await asyncio.create_subprocess_exec(
                    'ip', 'link', 'show',
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                stdout, _ = await result.communicate()
                
                if result.returncode == 0:
                    for line in stdout.decode().splitlines():
                        if ': ' in line and not line.startswith(' '):
                            interface = line.split(':')[1].strip().split('@')[0]
                            interfaces.append(interface)
            else:
                # Windows
                result = await asyncio.create_subprocess_exec(
                    'netsh', 'interface', 'show', 'interface',
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                stdout, _ = await result.communicate()
                
                if result.returncode == 0:
                    lines = stdout.decode().splitlines()
                    for line in lines[3:]:  # Skip header
                        if line.strip():
                            parts = line.split()
                            if len(parts) >= 4:
                                interface = ' '.join(parts[3:])
                                interfaces.append(interface)
        
        except Exception as e:
            self.logger.warning("Network interface detection failed", error=str(e))
            interfaces = ['lo', 'eth0']  # Default fallback
        
        return interfaces


class CrossPlatformFileHandler:
    """Handle file operations in a cross-platform manner."""
    
    def __init__(self, platform_info: PlatformInfo):
        self.platform_info = platform_info
        self.logger = structlog.get_logger("file_handler")
    
    def normalize_path(self, path: Union[str, Path]) -> Path:
        """Normalize path for current platform."""
        path = Path(path)
        
        # Handle Windows drive letters and UNC paths
        if self.platform_info.platform_type == PlatformType.WINDOWS:
            if str(path).startswith('//') or str(path).startswith('\\\\'):
                # UNC path - leave as is
                return path
            # Convert forward slashes to backslashes
            return Path(str(path).replace('/', '\\'))
        else:
            # Unix-like systems - convert backslashes to forward slashes
            return Path(str(path).replace('\\', '/'))
    
    async def ensure_directory(self, directory: Union[str, Path]) -> Path:
        """Ensure directory exists, creating it if necessary."""
        directory = self.normalize_path(directory)
        
        try:
            directory.mkdir(parents=True, exist_ok=True)
            
            # Set appropriate permissions on Unix-like systems
            if self.platform_info.is_unix_like():
                directory.chmod(0o755)
            
            return directory
            
        except Exception as e:
            self.logger.error("Failed to create directory",
                            directory=str(directory),
                            error=str(e))
            raise
    
    async def safe_write(self, file_path: Union[str, Path], content: Union[str, bytes]) -> None:
        """Write file safely with platform-appropriate line endings."""
        file_path = self.normalize_path(file_path)
        
        try:
            # Ensure parent directory exists
            await self.ensure_directory(file_path.parent)
            
            # Handle text content with proper line endings
            if isinstance(content, str):
                content = content.replace('\n', self.platform_info.line_separator)
                mode = 'w'
                encoding = 'utf-8'
            else:
                mode = 'wb'
                encoding = None
            
            # Atomic write using temporary file
            temp_file = file_path.with_suffix(file_path.suffix + '.tmp')
            
            if encoding:
                with open(temp_file, mode, encoding=encoding, newline='') as f:
                    f.write(content)
            else:
                with open(temp_file, mode) as f:
                    f.write(content)
            
            # Atomic rename
            temp_file.replace(file_path)
            
            # Set appropriate permissions on Unix-like systems
            if self.platform_info.is_unix_like():
                file_path.chmod(0o644)
            
            self.logger.debug("File written successfully", file_path=str(file_path))
            
        except Exception as e:
            self.logger.error("Failed to write file",
                            file_path=str(file_path),
                            error=str(e))
            # Clean up temp file if it exists
            temp_file = file_path.with_suffix(file_path.suffix + '.tmp')
            if temp_file.exists():
                temp_file.unlink()
            raise
    
    async def safe_read(self, file_path: Union[str, Path], binary: bool = False) -> Union[str, bytes]:
        """Read file safely with platform-appropriate handling."""
        file_path = self.normalize_path(file_path)
        
        try:
            if binary:
                with open(file_path, 'rb') as f:
                    return f.read()
            else:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # Normalize line endings
                    return content.replace('\r\n', '\n').replace('\r', '\n')
        
        except Exception as e:
            self.logger.error("Failed to read file",
                            file_path=str(file_path),
                            error=str(e))
            raise


class DependencyValidator:
    """Validate required dependencies and versions."""
    
    def __init__(self, platform_info: PlatformInfo):
        self.platform_info = platform_info
        self.logger = structlog.get_logger("dependency_validator")
    
    async def validate_python_dependencies(self) -> Dict[str, Any]:
        """Validate Python package dependencies."""
        required_packages = {
            'asyncio': None,  # Built-in, check version compatibility
            'pathlib': None,  # Built-in
            'json': None,     # Built-in
            'uuid': None,     # Built-in
            'hashlib': None,  # Built-in
            'ssl': None,      # Built-in
            'socket': None,   # Built-in
            # Optional dependencies
            'structlog': '>=20.0.0',
            'cryptography': '>=3.0.0',
            'pydantic': '>=1.8.0',
        }
        
        results = {}
        
        for package_name, version_req in required_packages.items():
            try:
                # Try to import the package
                spec = importlib.util.find_spec(package_name)
                
                if spec is None:
                    results[package_name] = {
                        'available': False,
                        'version': None,
                        'required': version_req,
                        'status': 'missing'
                    }
                    continue
                
                # Import the package to get version
                module = importlib.import_module(package_name)
                
                # Try to get version
                version = getattr(module, '__version__', 'unknown')
                
                results[package_name] = {
                    'available': True,
                    'version': version,
                    'required': version_req,
                    'status': 'available'
                }
                
            except Exception as e:
                results[package_name] = {
                    'available': False,
                    'version': None,
                    'required': version_req,
                    'status': f'error: {str(e)}'
                }
        
        # Check critical built-ins
        critical_missing = [
            name for name, info in results.items()
            if not info['available'] and name in ['asyncio', 'pathlib', 'json', 'uuid']
        ]
        
        if critical_missing:
            self.logger.error("Critical Python modules missing",
                            missing=critical_missing)
        
        return results
    
    async def validate_system_dependencies(self) -> Dict[str, Any]:
        """Validate system-level dependencies."""
        system_deps = {
            'git': 'Version control system',
            'curl': 'HTTP client (optional)',
            'openssl': 'Cryptography support',
        }
        
        results = {}
        
        for command, description in system_deps.items():
            try:
                # Check if command exists
                if shutil.which(command):
                    # Try to get version
                    result = await asyncio.create_subprocess_exec(
                        command, '--version',
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE
                    )
                    stdout, stderr = await result.communicate()
                    
                    version_output = stdout.decode() if stdout else stderr.decode()
                    version = version_output.split('\n')[0] if version_output else 'unknown'
                    
                    results[command] = {
                        'available': True,
                        'version': version,
                        'description': description,
                        'status': 'available'
                    }
                else:
                    results[command] = {
                        'available': False,
                        'version': None,
                        'description': description,
                        'status': 'missing'
                    }
            
            except Exception as e:
                results[command] = {
                    'available': False,
                    'version': None,
                    'description': description,
                    'status': f'error: {str(e)}'
                }
        
        return results
    
    async def validate_network_capabilities(self) -> Dict[str, Any]:
        """Validate network capabilities."""
        capabilities = {}
        
        # Test basic socket operations
        try:
            import socket
            
            # IPv4 support
            sock4 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock4.close()
            capabilities['ipv4'] = {'available': True, 'status': 'supported'}
        except:
            capabilities['ipv4'] = {'available': False, 'status': 'not supported'}
        
        # IPv6 support
        try:
            sock6 = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
            sock6.close()
            capabilities['ipv6'] = {'available': True, 'status': 'supported'}
        except:
            capabilities['ipv6'] = {'available': False, 'status': 'not supported'}
        
        # UDP support
        try:
            udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            udp_sock.close()
            capabilities['udp'] = {'available': True, 'status': 'supported'}
        except:
            capabilities['udp'] = {'available': False, 'status': 'not supported'}
        
        # SSL/TLS support
        try:
            import ssl
            context = ssl.create_default_context()
            capabilities['ssl'] = {
                'available': True,
                'version': ssl.OPENSSL_VERSION,
                'status': 'supported'
            }
        except:
            capabilities['ssl'] = {'available': False, 'status': 'not supported'}
        
        return capabilities


class PlatformCompatibilityManager:
    """Comprehensive platform compatibility management."""
    
    def __init__(self):
        self.logger = structlog.get_logger("compatibility_manager")
        self.detector = PlatformDetector()
        self.platform_info: Optional[PlatformInfo] = None
        self.file_handler: Optional[CrossPlatformFileHandler] = None
        self.dependency_validator: Optional[DependencyValidator] = None
    
    async def initialize(self) -> PlatformInfo:
        """Initialize platform compatibility system."""
        self.logger.info("Initializing platform compatibility system")
        
        # Detect platform
        self.platform_info = await self.detector.detect_platform()
        
        # Initialize components
        self.file_handler = CrossPlatformFileHandler(self.platform_info)
        self.dependency_validator = DependencyValidator(self.platform_info)
        
        # Validate compatibility
        if not self.platform_info.is_supported():
            self.logger.warning("Platform may not be fully supported",
                              platform=self.platform_info.platform_type.value,
                              architecture=self.platform_info.architecture.value,
                              python_version=str(self.platform_info.python_version))
        
        # Create necessary directories
        await self._ensure_platform_directories()
        
        self.logger.info("Platform compatibility system initialized successfully")
        return self.platform_info
    
    async def validate_environment(self) -> Dict[str, Any]:
        """Comprehensive environment validation."""
        if not self.platform_info or not self.dependency_validator:
            raise RuntimeError("Platform compatibility manager not initialized")
        
        self.logger.info("Starting comprehensive environment validation")
        
        validation_results = {
            'platform_info': {
                'platform': self.platform_info.platform_type.value,
                'architecture': self.platform_info.architecture.value,
                'python_version': '.'.join(map(str, self.platform_info.python_version)),
                'is_supported': self.platform_info.is_supported(),
                'container_type': self.platform_info.container_type.value,
                'memory_mb': self.platform_info.available_memory,
                'cpu_count': self.platform_info.cpu_count
            },
            'python_dependencies': await self.dependency_validator.validate_python_dependencies(),
            'system_dependencies': await self.dependency_validator.validate_system_dependencies(),
            'network_capabilities': await self.dependency_validator.validate_network_capabilities(),
            'directories': await self._validate_directories(),
            'overall_status': 'unknown'
        }
        
        # Calculate overall status
        issues = []
        
        # Check critical dependencies
        python_deps = validation_results['python_dependencies']
        critical_deps = ['asyncio', 'pathlib', 'json', 'uuid']
        missing_critical = [dep for dep in critical_deps 
                          if not python_deps.get(dep, {}).get('available', False)]
        
        if missing_critical:
            issues.append(f"Missing critical Python dependencies: {missing_critical}")
        
        # Check platform support
        if not self.platform_info.is_supported():
            issues.append("Platform may not be fully supported")
        
        # Check network capabilities
        network_caps = validation_results['network_capabilities']
        if not network_caps.get('ipv4', {}).get('available', False):
            issues.append("IPv4 networking not available")
        
        # Determine overall status
        if not issues:
            validation_results['overall_status'] = 'excellent'
        elif len(issues) <= 2:
            validation_results['overall_status'] = 'good'
        else:
            validation_results['overall_status'] = 'needs_attention'
        
        validation_results['issues'] = issues
        
        self.logger.info("Environment validation completed",
                        status=validation_results['overall_status'],
                        issues_count=len(issues))
        
        return validation_results
    
    async def _ensure_platform_directories(self) -> None:
        """Ensure all necessary platform directories exist."""
        if not self.platform_info or not self.file_handler:
            return
        
        directories = [
            self.platform_info.get_config_dir(),
            self.platform_info.get_data_dir(),
            self.platform_info.get_cache_dir()
        ]
        
        for directory in directories:
            await self.file_handler.ensure_directory(directory)
            self.logger.debug("Directory ensured", path=str(directory))
    
    async def _validate_directories(self) -> Dict[str, Any]:
        """Validate directory access and permissions."""
        if not self.platform_info:
            return {}
        
        directories = {
            'config': self.platform_info.get_config_dir(),
            'data': self.platform_info.get_data_dir(),
            'cache': self.platform_info.get_cache_dir()
        }
        
        results = {}
        
        for name, directory in directories.items():
            try:
                # Check if directory exists
                exists = directory.exists()
                
                # Check read/write permissions
                readable = os.access(directory, os.R_OK) if exists else False
                writable = os.access(directory, os.W_OK) if exists else False
                
                results[name] = {
                    'path': str(directory),
                    'exists': exists,
                    'readable': readable,
                    'writable': writable,
                    'status': 'ok' if (exists and readable and writable) else 'issues'
                }
                
            except Exception as e:
                results[name] = {
                    'path': str(directory),
                    'exists': False,
                    'readable': False,
                    'writable': False,
                    'status': f'error: {str(e)}'
                }
        
        return results


# Global platform compatibility manager instance
_platform_manager: Optional[PlatformCompatibilityManager] = None


async def get_platform_manager() -> PlatformCompatibilityManager:
    """Get or create the global platform compatibility manager."""
    global _platform_manager
    
    if _platform_manager is None:
        _platform_manager = PlatformCompatibilityManager()
        await _platform_manager.initialize()
    
    return _platform_manager


async def validate_platform_compatibility() -> bool:
    """Quick platform compatibility check."""
    try:
        manager = await get_platform_manager()
        validation_results = await manager.validate_environment()
        
        return validation_results['overall_status'] in ['excellent', 'good']
        
    except Exception as e:
        logger.error("Platform compatibility validation failed", error=str(e))
        return False