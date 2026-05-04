"""개발용 계정 seed 스크립트 — 1회 실행용"""

import asyncio

import bcrypt
from sqlalchemy import select

from core.dependencies import AsyncSessionLocal
from models.user import User


async def seed() -> None:
    async with AsyncSessionLocal() as session:
        admin_hash = bcrypt.hashpw("qwer1234".encode(), bcrypt.gensalt()).decode()
        admin_result = await session.execute(select(User).where(User.username == "admin"))
        admin = admin_result.scalar_one_or_none()
        if admin:
            admin.email = "admin@paperpilot.local"
            admin.password_hash = admin_hash
            admin.full_name = "Administrator"
            admin.is_active = True
            admin.is_admin = True
        else:
            session.add(
                User(
                    email="admin@paperpilot.local",
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

        result = await session.execute(select(User).where(User.email == "zmarzmzm@naver.com"))
        if result.scalar_one_or_none():
            await session.commit()
            print("관리자 계정 생성/갱신 완료: admin / qwer1234")
            print("테스트 계정은 이미 존재합니다. 스킵합니다.")
            return

        password_hash = bcrypt.hashpw("qwer1234".encode(), bcrypt.gensalt()).decode()

        user = User(
            email="zmarzmzm@naver.com",
            password_hash=password_hash,
            username="zmarzmar",
            full_name="민동명",
            affiliation="서경대학교",
            preferred_framework="pytorch",
            preferred_categories=[],
            is_active=True,
            is_admin=False,
        )
        session.add(user)
        await session.commit()
        print("관리자 계정 생성/갱신 완료: admin / qwer1234")
        print(f"테스트 계정 생성 완료: {user.email} (id={user.id})")


if __name__ == "__main__":
    asyncio.run(seed())
