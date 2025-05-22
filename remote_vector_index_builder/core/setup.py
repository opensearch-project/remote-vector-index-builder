# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

from setuptools import setup, find_namespace_packages

setup(
    name="remote-vector-index-builder-core",
    version="1.0.0",
    package_dir={"core": "."},
    packages=["core"] + ["core." + pkg for pkg in find_namespace_packages()],
)
