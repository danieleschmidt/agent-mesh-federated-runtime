#!/usr/bin/env python3
"""Health check script for Agent Mesh."""

import sys
import json
import time
from pathlib import Path

def check_health():
    """Check health of Agent Mesh service."""
    try:
        # Basic health check - verify config exists and is valid
        config_path = Path("configs/production.json")
        if not config_path.exists():
            print("❌ Configuration file not found")
            return False
            
        with open(config_path) as f:
            config = json.load(f)
            
        if config.get("app", {}).get("name") == "agent-mesh":
            print("✅ Agent Mesh configuration is valid")
            return True
        else:
            print("❌ Invalid configuration")
            return False
            
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False

if __name__ == "__main__":
    if check_health():
        sys.exit(0)
    else:
        sys.exit(1)
