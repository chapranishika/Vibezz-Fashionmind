"""
Auth routes: /auth/signup  /auth/login  /auth/me  /auth/logout
Uses bcrypt for password hashing, PyJWT for tokens.
"""
import os, uuid
from datetime import datetime, timedelta, timezone
from fastapi import APIRouter, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr
import bcrypt, jwt
from api.db import get_db

router = APIRouter(prefix="/auth", tags=["auth"])
bearer = HTTPBearer(auto_error=False)

_DEV_SECRET = "fashionmind-dev-secret-change-in-prod"
SECRET = os.getenv("JWT_SECRET", _DEV_SECRET)
if SECRET == _DEV_SECRET and os.getenv("ENVIRONMENT", "development").lower() != "development":
    raise RuntimeError(
        "JWT_SECRET is unset (or still the dev default) while ENVIRONMENT != development. "
        "Set a strong random JWT_SECRET before deploying."
    )
ALGO   = "HS256"
TTL    = 60 * 24 * 7   # 7 days in minutes

class SignupReq(BaseModel):
    email: str
    password: str
    name: str
    phone: str = None

class LoginReq(BaseModel):
    email: str
    password: str

def _make_token(user_id: str, email: str) -> str:
    exp = datetime.now(timezone.utc) + timedelta(minutes=TTL)
    # RFC 7519 requires `sub` to be a string; PyJWT >= 2.10 rejects non-str on decode.
    return jwt.encode({"sub": str(user_id), "email": email, "exp": exp}, SECRET, algorithm=ALGO)

def get_current_user(creds: HTTPAuthorizationCredentials = Depends(bearer)):
    if not creds:
        raise HTTPException(status_code=401, detail="Not authenticated")
    try:
        payload = jwt.decode(creds.credentials, SECRET, algorithms=[ALGO])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

@router.post("/signup")
def signup(req: SignupReq):
    db = get_db()
    # Check if email exists
    existing = db.table("user_auth").select("id").eq("email", req.email).execute()
    if existing.data:
        raise HTTPException(status_code=409, detail="Email already registered")
    # Hash password
    pw_hash = bcrypt.hashpw(req.password.encode(), bcrypt.gensalt()).decode()
    # Create customer record
    cid = f"FM{uuid.uuid4().hex[:12].upper()}"
    db.table("customers").insert({
        "customer_id": cid, "age": None, "age_bucket": "26-35",
        "club_member_status": "ACTIVE", "active": True,
        "engagement_score": 0.5, "age_norm": 0.3,
    }).execute()
    # Create auth record
    res = db.table("user_auth").insert({
        "email": req.email, "password_hash": pw_hash,
        "customer_id": cid, "name": req.name, "phone": req.phone,
    }).execute()
    user = res.data[0]
    token = _make_token(user["id"], req.email)
    return {"token": token, "customer_id": cid, "name": req.name, "email": req.email}

@router.post("/login")
def login(req: LoginReq):
    demo_email = os.getenv("DEMO_EMAIL")
    demo_pass = os.getenv("DEMO_PASSWORD")
    if demo_email and demo_pass and req.email == demo_email and req.password == demo_pass:
        demo_cid = os.getenv("DEMO_CUSTOMER_ID", "00009d946eec3ea54add5ba56d5210ea898def4b46c68570cf0096d962cacc75")
        token = _make_token("999999", demo_email)
        return {
            "token": token,
            "customer_id": demo_cid,
            "name": "Demo User",
            "email": demo_email
        }

    db = get_db()
    res = db.table("user_auth").select("*").eq("email", req.email).execute()
    if not res.data:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    user = res.data[0]
    if not bcrypt.checkpw(req.password.encode(), user["password_hash"].encode()):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    db.table("user_auth").update({"last_login": datetime.now(timezone.utc).isoformat()}).eq("id", user["id"]).execute()
    token = _make_token(user["id"], req.email)
    return {"token": token, "customer_id": user["customer_id"],
            "name": user["name"], "email": req.email}

@router.get("/me")
def me(user=Depends(get_current_user)):
    demo_email = os.getenv("DEMO_EMAIL")
    if demo_email and user.get("email") == demo_email:
        demo_cid = os.getenv("DEMO_CUSTOMER_ID", "00009d946eec3ea54add5ba56d5210ea898def4b46c68570cf0096d962cacc75")
        return {
            "email": demo_email,
            "name": "Demo User",
            "phone": "0000000000",
            "customer_id": demo_cid,
            "created_at": "2026-07-01T00:00:00Z"
        }
    db = get_db()
    res = db.table("user_auth").select("email,name,phone,customer_id,created_at").eq("id", user["sub"]).execute()
    if not res.data:
        raise HTTPException(status_code=404, detail="User not found")
    return res.data[0]


