# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.


def test_imports():
    try:
        from app import main
        from core.tasks import run_tasks
        import faiss

        print("All imports successful!")
        return 0
    except ImportError as e:
        print(f"Import failed: {e}")
        exit(1)


if __name__ == "__main__":
    test_imports()
