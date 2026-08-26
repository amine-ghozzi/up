# Alembic migration versions

Empty until the first revision is generated against a live Postgres (autogenerate needs a DB):

```bash
# with the compose datastores up (see docker-compose.yml):
FINALZE_DATABASE_URL=postgresql+asyncpg://finalze:finalze@localhost:5432/finalze \
  venv/Scripts/python -m alembic revision --autogenerate -m "initial schema"
FINALZE_DATABASE_URL=...  venv/Scripts/python -m alembic upgrade head
```

The ORM metadata (`src/api/models.py`) is the source of truth; `alembic/env.py` reads the
DSN from settings and uses `Base.metadata` as `target_metadata`.
