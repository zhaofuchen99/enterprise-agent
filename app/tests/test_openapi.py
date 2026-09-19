"""OpenAPI 契约检查（开发流程 6.2 门禁：所有公开接口有 Schema 与错误用例）。

这些是「文档即门禁」的用例：接口改了 Schema、加了参数、漏了错误用例，
这里会红，而不是等到前端联调时才发现。
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from fastapi import FastAPI

#: 本阶段对外的业务接口。
#:
#: **新增接口必须登记到这里**：门禁只扫这份清单，漏登记不会红——
#: 那个接口就静默地不受任何契约检查，而门禁看起来仍然全绿。
#: 实测踩过一次：`/tasks/{task_id}/trace` 交付时没登记，
#: 于是"每个公开接口都有 summary/tags/错误用例"这条约束对它不成立。
PUBLIC_PATHS = (
    "/api/auth/login",
    "/api/agent/chat",
    "/api/agent/tasks/{task_id}",
    "/api/agent/tasks/{task_id}/trace",
    "/api/agent/tasks/{task_id}/cancel",
    "/api/agent/tasks/{task_id}/stream",
    "/api/agent/tasks/{task_id}/stream-token",
)


@pytest.fixture
def schema(app: FastAPI) -> dict[str, Any]:
    return app.openapi()


def test_public_paths_are_documented(schema: dict[str, Any]) -> None:
    for path in PUBLIC_PATHS:
        assert path in schema["paths"], f"接口未出现在 OpenAPI 中：{path}"


@pytest.mark.parametrize("path", PUBLIC_PATHS)
def test_every_operation_has_summary_and_tags(schema: dict[str, Any], path: str) -> None:
    for method, operation in schema["paths"][path].items():
        assert operation.get("summary"), f"{method.upper()} {path} 缺 summary"
        assert operation.get("tags"), f"{method.upper()} {path} 缺 tags"


@pytest.mark.parametrize("path", PUBLIC_PATHS)
def test_every_operation_declares_error_responses(schema: dict[str, Any], path: str) -> None:
    """每个接口都要挂出**具体**的错误用例，且错误体指向统一的 ErrorResponse。"""
    for method, operation in schema["paths"][path].items():
        responses = operation["responses"]
        # 只数具体的错误状态码：4XX 是范围响应，单靠它会让「一个具名错误都没声明」
        # 蒙混过关；而 2XX 的响应体是 ApiResponse，不是这里的检查对象
        numeric_errors = sorted(code for code in responses if code.isdigit() and code[0] in "45")
        assert numeric_errors, f"{method.upper()} {path} 没有任何具名错误用例"

        for code in [*numeric_errors, "4XX"]:
            ref = responses[code]["content"]["application/json"]["schema"]["$ref"]
            assert ref.endswith("/ErrorResponse"), f"{path} {code} 的错误体不是 ErrorResponse"


@pytest.mark.parametrize("path", PUBLIC_PATHS)
def test_framework_validation_error_is_not_documented(schema: dict[str, Any], path: str) -> None:
    """422 + HTTPValidationError 必须被顶掉。

    参数校验失败在本项目里映射为 400 INVALID_ARGUMENT，FastAPI 自动补的那条
    422 描述的是不会发生的行为——文档里留一条错的比留空更误导人。
    """
    for operation in schema["paths"][path].values():
        assert "HTTPValidationError" not in json.dumps(operation["responses"], ensure_ascii=False)


def test_error_response_schema_lists_the_full_error_code_table(schema: dict[str, Any]) -> None:
    """错误码表出现在文档里，客户端可以据此生成枚举。"""
    from app.core.errors import ErrorCode

    error_schema = schema["components"]["schemas"]["ErrorResponse"]
    assert set(error_schema["required"]) == {"code", "message", "trace_id", "retryable"}
    # data 在错误响应里恒为 null，但键必须在——前端按同一套字段解析成功与失败
    assert "data" in error_schema["properties"]

    # pydantic 把枚举抽成了独立组件，顺着 $ref 取回来
    ref = error_schema["properties"]["code"]["$ref"]
    code_schema = schema["components"]["schemas"][ref.rsplit("/", 1)[-1]]
    assert set(code_schema["enum"]) == {code.value for code in ErrorCode}


@pytest.mark.parametrize(
    ("path", "model_name"),
    [
        ("/api/auth/login", "ApiResponse_LoginData_"),
        ("/api/agent/chat", "ApiResponse_TaskCreatedData_"),
        ("/api/agent/tasks/{task_id}", "ApiResponse_TaskDetailData_"),
    ],
)
def test_success_response_is_wrapped_in_envelope(
    schema: dict[str, Any], path: str, model_name: str
) -> None:
    """成功响应必须是统一外壳，而不是裸的 data 对象。"""
    assert model_name in schema["components"]["schemas"]

    response_schema = schema["paths"][path][
        "post" if path != "/api/agent/tasks/{task_id}" else "get"
    ]["responses"][
        "200" if path == "/api/auth/login" else "202" if path == "/api/agent/chat" else "200"
    ]

    ref = response_schema["content"]["application/json"]["schema"]["$ref"]
    assert ref.endswith(model_name)

    envelope = schema["components"]["schemas"][model_name]
    assert set(envelope["required"]) == {"code", "message", "trace_id"}


def test_idempotency_key_header_is_documented(schema: dict[str, Any]) -> None:
    """Idempotency-Key 是接口契约的一部分，OpenAPI 里必须能找到。"""
    parameters = schema["paths"]["/api/agent/chat"]["post"]["parameters"]
    names = {parameter["name"] for parameter in parameters}
    assert "Idempotency-Key" in names

    header = next(p for p in parameters if p["name"] == "Idempotency-Key")
    assert header["in"] == "header"
    assert header["required"] is False


def test_health_endpoints_are_tagged(schema: dict[str, Any]) -> None:
    assert "/health/live" in schema["paths"]
    assert "/health/ready" in schema["paths"]
    assert schema["paths"]["/health/live"]["get"]["tags"] == ["health"]
