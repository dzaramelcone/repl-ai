"""Tests for Session."""

import logging

from mahtab.session import Session
from mahtab.store import Store


def test_session_has_id():
    store = Store()
    session = Session(store)
    assert len(session.id) == 8  # hex[:8]


def test_session_has_empty_namespace():
    store = Store()
    session = Session(store)
    assert session.namespace == {}


def test_session_has_empty_messages():
    store = Store()
    session = Session(store)
    assert session.messages == []


def test_session_shares_store():
    store = Store()
    s1 = Session(store)
    s2 = Session(store)
    assert s1.store is s2.store


def test_session_exec_updates_namespace():
    store = Store()
    session = Session(store)
    session.exec("x = 42")
    assert session.namespace["x"] == 42


def test_session_exec_can_read_namespace():
    store = Store()
    session = Session(store)
    session.namespace["y"] = 10
    session.exec("z = y * 2")
    assert session.namespace["z"] == 20


def test_session_spawn_creates_child():
    store = Store()
    parent = Session(store)
    child = parent.spawn()
    assert child.parent is parent
    assert child in parent.children
    assert child.store is parent.store


def test_session_spawn_with_context():
    store = Store()
    parent = Session(store)
    child = parent.spawn(context={"query": "find bugs"})
    assert child.namespace["query"] == "find bugs"


def test_session_has_loggers():
    store = Store()
    session = Session(store)
    assert isinstance(session.log_user_repl, logging.Logger)
    assert isinstance(session.log_user_chat, logging.Logger)
    assert isinstance(session.log_llm_repl, logging.Logger)
    assert isinstance(session.log_llm_chat, logging.Logger)


def test_session_logger_names_include_id():
    store = Store()
    session = Session(store)
    assert session.id in session.log_user_repl.name
