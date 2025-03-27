# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.


def import_modules():
    try:
        from core.tasks import (
            build_index,
            create_vectors_dataset,
            upload_index,
            run_tasks,
            TaskResult,
        )
        from core.common.models import IndexBuildParameters

        print("All imports successful!")
        return 0
    except ImportError as e:
        print(f"Import failed: {e}")
        exit(1)


if __name__ == "__main__":
    import_modules()
