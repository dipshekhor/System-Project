# backend/app/routers/blog.py
from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from app.database import get_db
from app.models import UserProfile, BlogPost, BlogAnswer
from app.schemas import (
    BlogPostCreate, BlogAnswerCreate,
    BlogPostSummary, BlogPostDetail, BlogAnswerOut,
)
from app.routers.auth import decode_token

router = APIRouter()
_security = HTTPBearer()


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(_security),
    db: AsyncSession = Depends(get_db),
) -> UserProfile:
    user_id = decode_token(credentials.credentials)
    result = await db.execute(select(UserProfile).where(UserProfile.id == user_id))
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=404, detail="User not found.")
    return user


# ── GET /api/blog/posts ────────────────────────────────────────────────────────
@router.get("/blog/posts", response_model=list[BlogPostSummary])
async def list_posts(db: AsyncSession = Depends(get_db)):
    stmt = (
        select(
            BlogPost,
            UserProfile.name.label("author_name"),
            func.count(BlogAnswer.id).label("answer_count"),
        )
        .join(UserProfile, BlogPost.user_id == UserProfile.id)
        .outerjoin(BlogAnswer, BlogAnswer.post_id == BlogPost.id)
        .group_by(BlogPost.id, UserProfile.name)
        .order_by(BlogPost.created_at.desc())
    )
    rows = (await db.execute(stmt)).all()
    return [
        BlogPostSummary(
            id=post.id,
            user_id=post.user_id,
            author_name=author_name,
            title=post.title,
            answer_count=answer_count,
            created_at=post.created_at,
            updated_at=post.updated_at,
        )
        for post, author_name, answer_count in rows
    ]


# ── POST /api/blog/posts ───────────────────────────────────────────────────────
@router.post("/blog/posts", response_model=BlogPostSummary, status_code=201)
async def create_post(
    data: BlogPostCreate,
    current_user: UserProfile = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    post = BlogPost(user_id=current_user.id, title=data.title, body=data.body)
    db.add(post)
    await db.flush()
    await db.refresh(post)
    return BlogPostSummary(
        id=post.id,
        user_id=post.user_id,
        author_name=current_user.name,
        title=post.title,
        answer_count=0,
        created_at=post.created_at,
        updated_at=post.updated_at,
    )


# ── GET /api/blog/posts/{post_id} ──────────────────────────────────────────────
@router.get("/blog/posts/{post_id}", response_model=BlogPostDetail)
async def get_post(post_id: int, db: AsyncSession = Depends(get_db)):
    row = (
        await db.execute(
            select(BlogPost, UserProfile.name.label("author_name"))
            .join(UserProfile, BlogPost.user_id == UserProfile.id)
            .where(BlogPost.id == post_id)
        )
    ).first()
    if not row:
        raise HTTPException(status_code=404, detail="Post not found.")
    post, author_name = row

    answer_rows = (
        await db.execute(
            select(BlogAnswer, UserProfile.name.label("author_name"))
            .join(UserProfile, BlogAnswer.user_id == UserProfile.id)
            .where(BlogAnswer.post_id == post_id)
            .order_by(BlogAnswer.created_at.asc())
        )
    ).all()

    answers = [
        BlogAnswerOut(
            id=ans.id,
            post_id=ans.post_id,
            user_id=ans.user_id,
            author_name=ans_author,
            body=ans.body,
            created_at=ans.created_at,
        )
        for ans, ans_author in answer_rows
    ]

    return BlogPostDetail(
        id=post.id,
        user_id=post.user_id,
        author_name=author_name,
        title=post.title,
        body=post.body,
        answer_count=len(answers),
        answers=answers,
        created_at=post.created_at,
        updated_at=post.updated_at,
    )


# ── POST /api/blog/posts/{post_id}/answers ─────────────────────────────────────
@router.post("/blog/posts/{post_id}/answers", response_model=BlogAnswerOut, status_code=201)
async def create_answer(
    post_id: int,
    data: BlogAnswerCreate,
    current_user: UserProfile = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    post = (
        await db.execute(select(BlogPost).where(BlogPost.id == post_id))
    ).scalar_one_or_none()
    if not post:
        raise HTTPException(status_code=404, detail="Post not found.")

    answer = BlogAnswer(post_id=post_id, user_id=current_user.id, body=data.body)
    db.add(answer)
    await db.flush()
    await db.refresh(answer)
    return BlogAnswerOut(
        id=answer.id,
        post_id=answer.post_id,
        user_id=answer.user_id,
        author_name=current_user.name,
        body=answer.body,
        created_at=answer.created_at,
    )


# ── DELETE /api/blog/posts/{post_id} ──────────────────────────────────────────
@router.delete("/blog/posts/{post_id}", status_code=204)
async def delete_post(
    post_id: int,
    current_user: UserProfile = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    post = (
        await db.execute(select(BlogPost).where(BlogPost.id == post_id))
    ).scalar_one_or_none()
    if not post:
        raise HTTPException(status_code=404, detail="Post not found.")
    if post.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="You can only delete your own posts.")
    await db.delete(post)


# ── DELETE /api/blog/answers/{answer_id} ──────────────────────────────────────
@router.delete("/blog/answers/{answer_id}", status_code=204)
async def delete_answer(
    answer_id: int,
    current_user: UserProfile = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    answer = (
        await db.execute(select(BlogAnswer).where(BlogAnswer.id == answer_id))
    ).scalar_one_or_none()
    if not answer:
        raise HTTPException(status_code=404, detail="Answer not found.")
    if answer.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="You can only delete your own answers.")
    await db.delete(answer)
