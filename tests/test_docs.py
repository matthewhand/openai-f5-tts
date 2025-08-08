import pytest
import pytest
from app.server import app

@pytest.fixture
def client():
    return app.test_client()


def test_swagger_json(client):
    """Ensure the Swagger/OpenAPI JSON spec is served"""
    rv = client.get('/apidocs/swagger.json')
    assert rv.status_code == 200
    data = rv.get_json()
    assert isinstance(data, dict)
    assert 'swagger' in data or 'openapi' in data


def test_swagger_ui(client):
    """Ensure the Swagger UI HTML is accessible"""
    rv = client.get('/apidocs/')
    assert rv.status_code == 200
    assert 'text/html' in rv.content_type
