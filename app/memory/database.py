"""
Database engine and session management for PostgreSQL experience memory.
Supports PostgreSQL connections via DATABASE_URL and includes seamless SQLite fallback for testing environments.
"""
import os
import sys
from contextlib import contextmanager
from typing import Generator
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
from dotenv import load_dotenv

from app.memory.models import Base

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/shelter_memory")
FALLBACK_SQLITE_URL = f"sqlite:///{os.path.join(os.path.dirname(__file__), 'memory.db')}"

_engine = None
_SessionFactory = None
_is_postgres = False


def get_engine():
    """Create or return existing SQLAlchemy engine."""
    global _engine, _is_postgres
    if _engine is not None:
        return _engine

    if DATABASE_URL.startswith("postgresql"):
        try:
            engine = create_engine(
                DATABASE_URL,
                pool_pre_ping=True,
                connect_args={"connect_timeout": 3} if "psycopg2" in DATABASE_URL or "postgresql" in DATABASE_URL else {}
            )
            with engine.connect() as conn:
                pass
            _engine = engine
            _is_postgres = True
            return _engine
        except Exception:
            pass

    _engine = create_engine(
        FALLBACK_SQLITE_URL,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool
    )
    _is_postgres = False
    return _engine


def get_session_factory():
    """Get SQLAlchemy sessionmaker factory."""
    global _SessionFactory
    if _SessionFactory is None:
        engine = get_engine()
        _SessionFactory = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return _SessionFactory


@contextmanager
def get_db() -> Generator[Session, None, None]:
    """Provide a transactional database session scope with automatic commit and rollback."""
    SessionFactory = get_session_factory()
    session = SessionFactory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def init_db():
    """Initialize all tables defined in models.py."""
    engine = get_engine()
    Base.metadata.create_all(bind=engine)
    return True


def is_using_postgres() -> bool:
    """Check if the active engine is connected to a PostgreSQL instance."""
    get_engine()
    return _is_postgres
