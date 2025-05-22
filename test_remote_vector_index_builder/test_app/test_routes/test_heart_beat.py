# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from app.routes import heart_beat


@pytest.fixture
def app():
    app = FastAPI()
    app.include_router(heart_beat.router)
    return app


@pytest.fixture
def client(app):
    return TestClient(app)


def test_heart_beat(client):
    response = client.get("/_heart_beat")
    assert response.status_code == 200
