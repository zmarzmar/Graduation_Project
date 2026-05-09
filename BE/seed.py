"""개발용 계정 seed 스크립트 — 1회 실행용"""

import asyncio
import os

import bcrypt
from sqlalchemy import select

from core.dependencies import AsyncSessionLocal
from models.user import User

_ADMIN_EMAIL = os.getenv("SEED_ADMIN_EMAIL", "admin@paperpilot.local")
_ADMIN_PASSWORD = os.getenv("SEED_ADMIN_PASSWORD")
_TEST_EMAIL = os.getenv("SEED_TEST_EMAIL")
_TEST_PASSWORD = os.getenv("SEED_TEST_PASSWORD")
_TEST_USERNAME = os.getenv("SEED_TEST_USERNAME", "testuser")
_TEST_FULLNAME = os.getenv("SEED_TEST_FULLNAME", "Test User")
_TEST_AFFILIATION = os.getenv("SEED_TEST_AFFILIATION")


async def seed() -> None:
    async with AsyncSessionLocal() as session:
        if _ADMIN_PASSWORD:
            admin_hash = bcrypt.hashpw(_ADMIN_PASSWORD.encode(), bcrypt.gensalt()).decode()
            admin_result = await session.execute(select(User).where(User.username == "admin"))
            admin = admin_result.scalar_one_or_none()
            if admin:
                admin.email = _ADMIN_EMAIL
                admin.password_hash = admin_hash
                admin.full_name = "Administrator"
                admin.is_active = True
                admin.is_admin = True
            else:
                session.add(
                    User(
                        email=_ADMIN_EMAIL,
                        password_hash=admin_hash,
                        username="admin",
                        full_name="Administrator",
                        affiliation=None,
                        preferred_framework="pytorch",
                        preferred_categories=[],
                        is_active=True,
                        is_admin=True,
                    )
                )
            print("관리자 계정 생성/갱신 완료: admin")
        else:
            print("SEED_ADMIN_PASSWORD 미설정 — 관리자 계정 생성/갱신 스킵")

        # 테스트 유저 — SEED_TEST_EMAIL, SEED_TEST_PASSWORD 환경변수 미설정 시 스킵
        if not _TEST_EMAIL or not _TEST_PASSWORD:
            await session.commit()
            print("SEED_TEST_EMAIL / SEED_TEST_PASSWORD 미설정 — 테스트 유저 생성 스킵")
            return

        result = await session.execute(select(User).where(User.email == _TEST_EMAIL))
        if result.scalar_one_or_none():
            await session.commit()
            print("테스트 계정은 이미 존재합니다. 스킵합니다.")
            return

        password_hash = bcrypt.hashpw(_TEST_PASSWORD.encode(), bcrypt.gensalt()).decode()

        user = User(
            email=_TEST_EMAIL,
            password_hash=password_hash,
            username=_TEST_USERNAME,
            full_name=_TEST_FULLNAME,
            affiliation=_TEST_AFFILIATION,
            preferred_framework="pytorch",
            preferred_categories=[],
            is_active=True,
            is_admin=False,
        )
        session.add(user)
        await session.commit()
        print(f"테스트 계정 생성 완료: {user.email} (id={user.id})")


if __name__ == "__main__":
    asyncio.run(seed())
