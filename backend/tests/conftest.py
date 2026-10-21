import os
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.compiler import compiles

# Set before any test module is imported (their own setdefault calls then no-op),
# so signing session tokens uses a key of a realistic length instead of
# triggering PyJWT's short-HMAC-key warning.
os.environ.setdefault("SECRET_KEY", "test-secret-key-that-is-at-least-32-bytes-long")


@compiles(JSONB, "sqlite")
def compile_jsonb_sqlite(type_, compiler, **kw):
    return "JSON"
