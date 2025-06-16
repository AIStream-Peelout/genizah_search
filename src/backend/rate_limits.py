import os
import redis
import hashlib
from datetime import datetime
from typing import Optional
from fastapi import Request
from pydantic import BaseModel
from typing import List
import logging
logger = logging.getLogger(__name__)
import logging

class RateLimitExceeded(Exception):
    def __init__(self, message: str):
        self.message = message

class UsageStats(BaseModel):
    global_queries_today: int
    global_limit: int
    your_queries_hour: int
    hourly_limit: int
    your_queries_today: int
    daily_limit: int
    estimated_cost_today: float
    budget_cap: float
    remaining_queries_today: int

class FilterOptions(BaseModel):
    languages: List[str]
    periods: List[str]
    document_types: List[str]
    institutions: List[str]
    collections: List[str]

class ProtectionService:
    """Rate limiting and cost protection service"""

    def __init__(self):
        # Configuration
        self.DAILY_QUERY_LIMIT = int(os.getenv('DAILY_QUERY_LIMIT', 1000))
        self.USER_HOURLY_LIMIT = int(os.getenv('USER_HOURLY_LIMIT', 10))
        self.USER_DAILY_LIMIT = int(os.getenv('USER_DAILY_LIMIT', 50))
        self.QUERY_COST = float(os.getenv('QUERY_COST', 0.0002))
        self.DAILY_BUDGET_CAP = float(os.getenv('DAILY_BUDGET_CAP', 0.20))

        # Redis connection
        self.redis_client = None
        self.in_memory_store = {}
        self._init_redis()

    def _init_redis(self):
        """Initialize Redis connection with fallback to in-memory"""
        try:
            redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379')
            self.redis_client = redis.from_url(redis_url, decode_responses=True)
            self.redis_client.ping()
            logger.info("Connected to Redis for rate limiting")
        except Exception as e:
            logger.warning(f"Redis not available, using in-memory storage: {e}")
            self.redis_client = None

    def _get_client_id(self, request: Request) -> str:
        """Generate client identifier from request"""
        forwarded_for = request.headers.get('x-forwarded-for')
        client_ip = forwarded_for.split(',')[0] if forwarded_for else request.client.host
        user_agent = request.headers.get('user-agent', '')
        client_string = f"{client_ip}:{user_agent}"
        return hashlib.md5(client_string.encode()).hexdigest()[:12]

    def _get_key(self, client_id: Optional[str], window: str) -> str:
        """Generate storage key"""
        today = datetime.now().strftime('%Y-%m-%d')
        hour = datetime.now().strftime('%Y-%m-%d-%H')

        if window == 'global_daily':
            return f"genizah:global:{today}"
        elif window == 'daily':
            return f"genizah:daily:{today}:{client_id}"
        elif window == 'hourly':
            return f"genizah:hourly:{hour}:{client_id}"

    def _get_count(self, key: str) -> int:
        """Get current count for key"""
        if self.redis_client:
            try:
                count = self.redis_client.get(key)
                return int(count) if count else 0
            except:
                logger.error("Redis error, falling back to in-memory")
                return self.in_memory_store.get(key, 0)
        else:
            return self.in_memory_store.get(key, 0)

    def _increment_count(self, key: str, ttl_seconds: int) -> int:
        """Increment count with TTL"""
        if self.redis_client:
            try:
                pipe = self.redis_client.pipeline()
                pipe.incr(key)
                pipe.expire(key, ttl_seconds)
                result = pipe.execute()
                return result[0]
            except:
                logger.error("Redis error, falling back to in-memory")
                self.in_memory_store[key] = self.in_memory_store.get(key, 0) + 1
                return self.in_memory_store[key]
        else:
            self.in_memory_store[key] = self.in_memory_store.get(key, 0) + 1
            return self.in_memory_store[key]

    async def check_limits(self, request: Request) -> None:
        """Check if request is within limits, raise exception if not"""
        client_id = self._get_client_id(request)

        # Check global daily limit
        global_key = self._get_key(None, 'global_daily')
        global_count = self._get_count(global_key)

        if global_count >= self.DAILY_QUERY_LIMIT:
            raise RateLimitExceeded(
                f"Demo has reached daily limit of {self.DAILY_QUERY_LIMIT} queries. Try again tomorrow!"
            )

        # Check global budget
        global_cost = global_count * self.QUERY_COST
        if global_cost >= self.DAILY_BUDGET_CAP:
            raise RateLimitExceeded(
                f"Demo has reached daily budget cap of ${self.DAILY_BUDGET_CAP:.2f}. Try again tomorrow!"
            )

        # Check user hourly limit
        hourly_key = self._get_key(client_id, 'hourly')
        hourly_count = self._get_count(hourly_key)

        if hourly_count >= self.USER_HOURLY_LIMIT:
            raise RateLimitExceeded(
                f"You've reached the hourly limit of {self.USER_HOURLY_LIMIT} queries. Please wait."
            )

        # Check user daily limit
        daily_key = self._get_key(client_id, 'daily')
        daily_count = self._get_count(daily_key)

        if daily_count >= self.USER_DAILY_LIMIT:
            raise RateLimitExceeded(
                f"You've reached the daily limit of {self.USER_DAILY_LIMIT} queries. Try again tomorrow!"
            )

    async def record_query(self, request: Request) -> None:
        """Record a query and update all counters"""
        client_id = self._get_client_id(request)

        # Increment all counters
        global_key = self._get_key(None, 'global_daily')
        hourly_key = self._get_key(client_id, 'hourly')
        daily_key = self._get_key(client_id, 'daily')

        self._increment_count(global_key, 86400)  # 24 hours
        self._increment_count(hourly_key, 3600)  # 1 hour
        self._increment_count(daily_key, 86400)  # 24 hours

        logger.info(f"Query recorded for client {client_id}")

    async def get_usage_stats(self, request: Request) -> UsageStats:
        """Get current usage statistics"""
        client_id = self._get_client_id(request)

        global_key = self._get_key(None, 'global_daily')
        hourly_key = self._get_key(client_id, 'hourly')
        daily_key = self._get_key(client_id, 'daily')

        global_count = self._get_count(global_key)
        hourly_count = self._get_count(hourly_key)
        daily_count = self._get_count(daily_key)

        return UsageStats(
            global_queries_today=global_count,
            global_limit=self.DAILY_QUERY_LIMIT,
            your_queries_hour=hourly_count,
            hourly_limit=self.USER_HOURLY_LIMIT,
            your_queries_today=daily_count,
            daily_limit=self.USER_DAILY_LIMIT,
            estimated_cost_today=global_count * self.QUERY_COST,
            budget_cap=self.DAILY_BUDGET_CAP,
            remaining_queries_today=max(0, self.DAILY_QUERY_LIMIT - global_count)
        )