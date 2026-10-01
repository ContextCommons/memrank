"""An httpx transport that hands each request to an ASGI app in process -- no socket, no clock."""

from __future__ import annotations

import httpx
from fastapi import FastAPI
from fastapi.testclient import TestClient


def asgi_transport(app: FastAPI) -> httpx.MockTransport:
    client = TestClient(app)

    def handle(request: httpx.Request) -> httpx.Response:
        response = client.request(request.method, request.url.raw_path.decode(),
                                  content=request.content,
                                  headers={"content-type": "application/json"})
        return httpx.Response(response.status_code, content=response.content,
                              headers={"content-type": response.headers.get(
                                  "content-type", "application/json")})

    return httpx.MockTransport(handle)
