#!/usr/bin/env python3
"""Alias entry point - the kit accepts serve.py or harness.py; ours is serve.py."""
from serve import main
import asyncio

if __name__ == "__main__":
    asyncio.run(main())
