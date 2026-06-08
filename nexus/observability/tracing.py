"""
nexus/observability/tracing.py
-------------------------------
OpenTelemetry setup that sends all traces to local Jaeger.
Every agent action, LLM call, graph query, and tool execution
gets a span — giving you a live map of what the kernel is doing.

Usage (call once at startup):
    from nexus.observability.tracing import setup_tracing, get_tracer
    setup_tracing()
    tracer = get_tracer("nexus.orchestration")

    with tracer.start_as_current_span("plan_step") as span:
        span.set_attribute("step_id", 42)
        ...
"""

from __future__ import annotations

import functools
from contextlib import contextmanager
from typing import Any, Callable, Generator

import structlog
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.sdk.trace.sampling import TraceIdRatioBased

from nexus.config.settings import get_settings

log = structlog.get_logger(__name__)

_provider: TracerProvider | None = None


def setup_tracing() -> TracerProvider:
    """
    Initialise the global OpenTelemetry TracerProvider.
    Call ONCE at application startup (main.py / cli.py).

    Sends spans to Jaeger via OTLP gRPC on localhost:4317.
    Falls back to console export if Jaeger is unreachable.
    """
    global _provider
    if _provider is not None:
        return _provider  # idempotent

    cfg = get_settings().observability

    resource = Resource.create(
        {
            "service.name": cfg.service_name,
            "service.version": get_settings().kernel_version,
            "deployment.environment": get_settings().environment,
        }
    )

    sampler = TraceIdRatioBased(cfg.sample_rate)
    _provider = TracerProvider(resource=resource, sampler=sampler)

    if cfg.enabled:
        try:
            otlp_exporter = OTLPSpanExporter(
                endpoint=cfg.otlp_endpoint,
                insecure=True,  # local Jaeger, no TLS needed
            )
            _provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
            log.info(
                "OpenTelemetry → Jaeger connected",
                endpoint=cfg.otlp_endpoint,
                service=cfg.service_name,
            )
        except Exception as exc:
            # Don't crash the kernel if Jaeger isn't up yet
            log.warning(
                "Jaeger OTLP export failed, falling back to console",
                error=str(exc),
            )
            _provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
    else:
        log.info("Tracing disabled — using console exporter")
        _provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))

    trace.set_tracer_provider(_provider)
    return _provider


def get_tracer(name: str) -> trace.Tracer:
    """
    Get a named tracer. Each module should use its own tracer name
    for easy filtering in the Jaeger UI.

    Example names:
        "nexus.orchestration.planner"
        "nexus.epistemic.graph"
        "nexus.mcp_fabric.synthesizer"
    """
    if _provider is None:
        setup_tracing()
    return trace.get_tracer(name)


@contextmanager
def traced_span(
    tracer_name: str,
    span_name: str,
    attributes: dict[str, Any] | None = None,
) -> Generator[trace.Span, None, None]:
    """
    Context manager that creates a span, sets attributes, and
    records exceptions automatically.

    Usage:
        with traced_span("nexus.agents", "execute_tool", {"tool": "postgres_query"}):
            result = await tool.run(query)
    """
    tracer = get_tracer(tracer_name)
    with tracer.start_as_current_span(span_name) as span:
        if attributes:
            for k, v in attributes.items():
                span.set_attribute(k, str(v))
        try:
            yield span
        except Exception as exc:
            span.record_exception(exc)
            span.set_status(trace.StatusCode.ERROR, str(exc))
            raise


def traced(tracer_name: str, span_name: str | None = None) -> Callable:
    """
    Decorator that wraps an async function with a traced span.

    Usage:
        @traced("nexus.epistemic", "graph_query")
        async def query_graph(self, cypher: str) -> list[dict]:
            ...
    """
    def decorator(func: Callable) -> Callable:
        name = span_name or func.__qualname__

        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            with traced_span(tracer_name, name):
                return await func(*args, **kwargs)

        return wrapper

    return decorator
