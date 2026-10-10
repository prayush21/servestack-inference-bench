"""Minimal Server-Sent Events parsing shared by both backends."""

from __future__ import annotations

from collections.abc import AsyncIterable, AsyncIterator, Iterable, Iterator


class _EventBuilder:
    def __init__(self) -> None:
        self.data: list[str] = []

    def feed(self, line: str) -> str | None:
        """Feed one line; return a complete event's data when a blank line ends it."""
        line = line.rstrip("\r\n")
        if line == "":
            if not self.data:
                return None
            event, self.data = "\n".join(self.data), []
            return event
        if line.startswith(":"):  # comment / keep-alive
            return None
        name, _, value = line.partition(":")
        if name == "data":
            self.data.append(value[1:] if value.startswith(" ") else value)
        return None

    def flush(self) -> str | None:
        if not self.data:
            return None
        event, self.data = "\n".join(self.data), []
        return event


def iter_sse_data(lines: Iterable[str]) -> Iterator[str]:
    """Yield the `data` payload of each SSE event. A trailing event without a blank line is
    still yielded, since some servers close the stream right after the last `data:` line."""
    builder = _EventBuilder()
    for line in lines:
        event = builder.feed(line)
        if event is not None:
            yield event
    event = builder.flush()
    if event is not None:
        yield event


async def aiter_sse_data(lines: AsyncIterable[str]) -> AsyncIterator[str]:
    builder = _EventBuilder()
    async for line in lines:
        event = builder.feed(line)
        if event is not None:
            yield event
    event = builder.flush()
    if event is not None:
        yield event
