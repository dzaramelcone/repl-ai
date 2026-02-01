"""Tests for Store."""

from mahtab.store import Store


def test_store_starts_empty():
    store = Store()
    assert store.data == b""


def test_store_append_bytes():
    store = Store()
    store.append(b"hello")
    assert store.data == b"hello"


def test_store_append_str():
    store = Store()
    store.append("hello")
    assert store.data == b"hello"


def test_store_append_accumulates():
    store = Store()
    store.append(b"hello")
    store.append(b" world")
    assert store.data == b"hello world"


def test_store_load_full():
    store = Store()
    store.append(b"hello world")
    assert store.load() == b"hello world"


def test_store_load_slice():
    store = Store()
    store.append(b"hello world")
    assert store.load(0, 5) == b"hello"
    assert store.load(6, 11) == b"world"


def test_store_load_start_only():
    store = Store()
    store.append(b"hello world")
    assert store.load(6) == b"world"
