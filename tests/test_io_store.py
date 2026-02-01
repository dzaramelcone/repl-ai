"""Tests for memory store."""

from mahtab.io.store import MemoryStore


def test_memory_store_append():
    store = MemoryStore()
    store.append(b"hello")
    store.append(b" world")
    assert store.data == b"hello world"


def test_memory_store_clear():
    store = MemoryStore()
    store.append(b"data")
    store.clear()
    assert store.data == b""
