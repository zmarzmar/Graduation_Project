"""테스트 계정 seed 스크립트 — 1회 실행용"""

import asyncio

import bcrypt
from sqlalchemy import select

from core.dependencies import AsyncSessionLocal
from models.user import User


async def seed() -> None:
    async with AsyncSessionLocal() as session:
        # 이미 존재하면 스킵
        result = await session.execute(select(User).where(User.email == "zmarzmzm@naver.com"))
        if result.scalar_one_or_none():
            print("이미 존재하는 계정입니다. 스킵합니다.")
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
        )
        session.add(user)
        await session.commit()
        print(f"테스트 계정 생성 완료: {user.email} (id={user.id})")


if __name__ == "__main__":
    asyncio.run(seed())
