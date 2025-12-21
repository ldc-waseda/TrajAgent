"""
OpenAI Client Wrapper with transparent cache and batch support.

This wrapper intercepts OpenAI API calls and routes them based on mode:
- realtime: Normal API call (default)
- batch_write: Save request to batch queue, return cache or error
- cache_first: Return cached response or save to batch queue

The wrapper is transparent - existing code doesn't need modification.
"""

import os
import json
import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types.completion_usage import CompletionUsage

from .cache_manager import CacheManager


class BatchQueueManager:
    """Manages the queue of requests to be batched."""
    
    def __init__(self, cache_dir: str = ".cache", task_name: Optional[str] = None):
        """
        Initialize batch queue manager.
        
        Args:
            cache_dir: Base cache directory
            task_name: Task directory name (e.g., "gpt-4o_20250104_143022")
                      If None, uses default "default" directory
        """
        self.cache_dir = Path(cache_dir)
        self.queue_base_dir = self.cache_dir / "batch_queue"
        self.queue_base_dir.mkdir(parents=True, exist_ok=True)
        
        # Use task-specific directory
        self.task_name = task_name or "default"
        self.queue_dir = self.queue_base_dir / self.task_name
        self.queue_dir.mkdir(parents=True, exist_ok=True)
        
        # Task info file
        self.task_info_file = self.queue_dir / "task_info.json"
    
    def create_task_info(self, model: str, strategy: str = "", description: str = ""):
        """
        Create or update task info file.
        
        Args:
            model: Model name
            strategy: Generation strategy
            description: Task description
        """
        task_info = {
            "task_name": self.task_name,
            "model": model,
            "strategy": strategy,
            "description": description,
            "created_at": datetime.now().isoformat(),
            "total_requests": 0
        }
        
        # Update if exists
        if self.task_info_file.exists():
            try:
                with open(self.task_info_file, 'r', encoding='utf-8') as f:
                    existing = json.load(f)
                    task_info["created_at"] = existing.get("created_at", task_info["created_at"])
            except Exception:
                pass
        
        with open(self.task_info_file, 'w', encoding='utf-8') as f:
            json.dump(task_info, f, ensure_ascii=False, indent=2)
    
    def get_task_info(self) -> Optional[Dict[str, Any]]:
        """Get task info."""
        if self.task_info_file.exists():
            try:
                with open(self.task_info_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                pass
        return None
    
    def add_request(
        self,
        request_hash: str,
        messages: List[Dict[str, Any]],
        model: str,
        temperature: float,
        max_completion_tokens: int,
        response_format: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> str:
        """
        Add a request to the batch queue.
        
        Returns:
            Path to the queued request file
        """
        request_data = {
            "hash": request_hash,
            "messages": messages,
            "model": model,
            "temperature": temperature,
            "max_completion_tokens": max_completion_tokens,
            "created_at": datetime.now().isoformat(),
            "task_name": self.task_name,
            **kwargs
        }
        
        if response_format:
            request_data["response_format"] = response_format
        
        # Save to queue
        queue_file = self.queue_dir / f"{request_hash}.json"
        with open(queue_file, 'w', encoding='utf-8') as f:
            json.dump(request_data, f, ensure_ascii=False, indent=2)
        
        # Update task info
        task_info = self.get_task_info()
        if task_info:
            task_info["total_requests"] = self.count_queued()
            task_info["last_updated"] = datetime.now().isoformat()
            with open(self.task_info_file, 'w', encoding='utf-8') as f:
                json.dump(task_info, f, ensure_ascii=False, indent=2)
        
        return str(queue_file)
    
    def get_queued_requests(self, sort_by_scenario: bool = True) -> List[Dict[str, Any]]:
        """
        Get all queued requests.
        
        Args:
            sort_by_scenario: If True, sort requests by scenario and idx for better cache locality
            
        Returns:
            List of request dictionaries
        """
        requests = []
        for file_path in self.queue_dir.glob("*.json"):
            # Skip task_info.json
            if file_path.name == "task_info.json":
                continue
                
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    request = json.load(f)
                    requests.append(request)
            except (json.JSONDecodeError, OSError) as e:
                print(f"[Queue] Failed to load {file_path}: {e}")
        
        if sort_by_scenario and requests:
            # Sort by scenario (primary), then by idx (secondary) for cache locality
            # Use created_at as fallback if scenario/idx not available
            def sort_key(req):
                scenario = req.get("scenario", "")
                idx = req.get("idx", 0)
                created_at = req.get("created_at", "")
                return (scenario, idx, created_at)
            
            requests.sort(key=sort_key)
            print(f"[Queue] Sorted {len(requests)} requests by scenario and idx for cache locality")
        
        return requests
    
    def clear_request(self, request_hash: str) -> bool:
        """Remove a request from the queue."""
        queue_file = self.queue_dir / f"{request_hash}.json"
        if queue_file.exists():
            queue_file.unlink()
            return True
        return False
    
    def count_queued(self) -> int:
        """Count queued requests (excluding task_info.json)."""
        all_files = list(self.queue_dir.glob("*.json"))
        # Exclude task_info.json
        return len([f for f in all_files if f.name != "task_info.json"])
    
    @staticmethod
    def list_tasks(cache_dir: str = ".cache") -> List[str]:
        """
        List all task directories.
        
        Returns:
            List of task names
        """
        queue_base_dir = Path(cache_dir) / "batch_queue"
        if not queue_base_dir.exists():
            return []
        
        tasks = []
        for item in queue_base_dir.iterdir():
            if item.is_dir():
                tasks.append(item.name)
        return sorted(tasks)
    
    @staticmethod
    def get_task_summary(cache_dir: str = ".cache", task_name: str = "") -> Optional[Dict[str, Any]]:
        """
        Get summary of a task.
        
        Args:
            cache_dir: Base cache directory
            task_name: Task directory name
            
        Returns:
            Task summary dictionary
        """
        task_dir = Path(cache_dir) / "batch_queue" / task_name
        task_info_file = task_dir / "task_info.json"
        
        if not task_info_file.exists():
            return None
        
        try:
            with open(task_info_file, 'r', encoding='utf-8') as f:
                task_info = json.load(f)
            
            # Count files
            queue_count = len([f for f in task_dir.glob("*.json") if f.name != "task_info.json"])
            task_info["queued_requests"] = queue_count
            
            return task_info
        except Exception:
            return None


class MockChatCompletion:
    """Mock ChatCompletion response for cache hits."""
    
    def __init__(self, content: str, model: str = "cached"):
        self.id = "cached"
        self.model = model
        self.object = "chat.completion"
        self.created = int(datetime.now().timestamp())
        
        message = ChatCompletionMessage(
            role="assistant",
            content=content
        )
        message.parsed = None  # Will be set by wrapper
        
        self.choices = [
            Choice(
                index=0,
                message=message,
                finish_reason="stop"
            )
        ]
        
        self.usage = CompletionUsage(
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0
        )


class WrappedChatCompletions:
    """Wrapper for chat.completions with cache and batch support."""
    
    def __init__(
        self, 
        original_completions: Any,
        mode: str,
        cache_mgr: CacheManager,
        queue_mgr: BatchQueueManager,
        model: str = "",
        semaphore: Optional[asyncio.Semaphore] = None
    ):
        self._original = original_completions
        self.mode = mode
        self.cache_mgr = cache_mgr
        self.queue_mgr = queue_mgr
        self.model = model
        self.semaphore = semaphore
    
    async def parse(
        self,
        *,
        messages: List[Dict[str, Any]],
        model: str,
        temperature: float = 0.2,
        max_completion_tokens: int = 1024,
        response_format: Optional[Any] = None,
        **kwargs
    ) -> Any:
        """
        Intercept chat.completions.parse() call.
        
        Routes based on mode:
        - realtime: Call actual API
        - batch_write: Save to queue, check cache first
        - cache_first: Check cache, save to queue if miss
        """
        # Compability rewrite
        reasoning_effort = None
        if "reasoning_effort" in kwargs:
            reasoning_effort = kwargs["reasoning_effort"]
            if model.startswith("Qwen/"):
                from sglang.srt.sampling.custom_logit_processor import Qwen3ThinkingBudgetLogitProcessor
                # kwargs["reasoning_effort"] = "low"
                kwargs.pop("reasoning_effort")
                # kwargs["chat_template_kwargs"] = {
                #     "enable_thinking": "true"
                # }
                kwargs["extra_body"] = {
                    "custom_logit_processor": Qwen3ThinkingBudgetLogitProcessor().to_str(),
                    "custom_params": {
                        "thinking_budget": 2048,
                    },
                }
        
        # Extract response_format info for hashing and storage
        response_format_dict = None
        response_format_class = None
        if response_format is not None:
            response_format_class = response_format
            try:
                # Convert Pydantic model to dict for storage
                response_format_dict = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": response_format.__name__,
                        "schema": response_format.model_json_schema()
                    }
                }
            except (AttributeError, TypeError):
                response_format_dict = None
        
        # Compute request hash
        request_hash = self.cache_mgr.compute_request_hash(
            messages, model, temperature, max_completion_tokens,
            reasoning_effort=reasoning_effort
        )
        
        # Check cache first
        cached_response = self.cache_mgr.get_cached_response(request_hash)
        
        if self.mode == "realtime":
            # Realtime mode: check cache, then call API
            if cached_response is not None:
                print(f"[Cache] Hit for hash {request_hash[:8]}...")
                return self._build_response_from_cache(cached_response, response_format_class)
            
            # Call actual API with concurrency limit
            if self.semaphore is not None:
                async with self.semaphore:
                    resp = await self._original.parse(
                        messages=messages,
                        model=model,
                        temperature=temperature,
                        max_completion_tokens=max_completion_tokens,
                        response_format=response_format,
                        **kwargs
                    )
            else:
                resp = await self._original.parse(
                    messages=messages,
                    model=model,
                    temperature=temperature,
                    max_completion_tokens=max_completion_tokens,
                    response_format=response_format,
                    **kwargs
                )
            
            # Save to cache
            try:
                response_data = self._extract_response_data(resp)
                self.cache_mgr.save_response(request_hash, response_data)
                self.cache_mgr.save_request(
                    request_hash, messages, model, temperature, max_completion_tokens,
                    response_format=response_format_dict
                )
            except (OSError, KeyError, ValueError) as e:
                print(f"[Cache] Failed to save response: {e}")
            
            return resp
        
        elif self.mode == "batch_write":
            # Batch write mode: check cache first, then queue
            if cached_response is not None:
                print(f"[Cache] Hit for hash {request_hash[:8]}...")
                return self._build_response_from_cache(cached_response, response_format_class)
            
            # Add to batch queue
            self.queue_mgr.add_request(
                request_hash, messages, model, temperature, max_completion_tokens,
                response_format=response_format_dict, **kwargs
            )
            print(f"[Batch] Queued request {request_hash[:8]}...")
            
            # Raise an error to signal that this request is queued
            raise RuntimeError(f"REQUEST_QUEUED: {request_hash}")
        
        elif self.mode == "cache_first":
            # Cache first mode: only use cache
            if cached_response is not None:
                print(f"[Cache] Hit for hash {request_hash[:8]}...")
                return self._build_response_from_cache(cached_response, response_format_class)
            
            # No cache, add to queue and raise error
            self.queue_mgr.add_request(
                request_hash, messages, model, temperature, max_completion_tokens,
                response_format=response_format_dict, **kwargs
            )
            print(f"[Cache] Miss for hash {request_hash[:8]}, queued for batch")
            
            raise RuntimeError(f"CACHE_MISS: {request_hash}")
        
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
    
    def _extract_response_data(self, resp: Any) -> Dict[str, Any]:
        """Extract response data for caching."""
        if not hasattr(resp, 'choices') or not resp.choices:
            return {}
        
        choice = resp.choices[0]
        message = choice.message
        
        return {
            "id": resp.id,
            "model": resp.model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": message.role,
                    "content": message.content,
                    "reasoning_content": message.reasoning_content if hasattr(message, "reasoning_content") else None,
                    "tool_calls": message.tool_calls if hasattr(message, "tool_calls") else None,
                },
                "finish_reason": choice.finish_reason
            }],
            # "usage": resp.usage if hasattr(resp, "usage") else None,
        }
    
    def _build_response_from_cache(
        self, 
        cached_data: Dict[str, Any],
        response_format_class: Optional[Any] = None
    ) -> Any:
        """Build a mock response from cached data."""
        # Extract content from cached response
        response_body = cached_data.get("response", cached_data)
        choices = response_body.get("choices", [])
        
        if not choices:
            raise ValueError("No choices in cached response")
        
        content = choices[0].get("message", {}).get("content", "")
        model = response_body.get("model", "cached")
        
        # Create mock response
        mock_resp = MockChatCompletion(content, model)
        
        # Parse content if response format is provided
        if response_format_class and content:
            try:
                parsed_data = json.loads(content)
                mock_resp.choices[0].message.parsed = response_format_class(**parsed_data)
            except (json.JSONDecodeError, TypeError, ValueError) as e:
                print(f"[Cache] Failed to parse cached content: {e}")
                mock_resp.choices[0].message.parsed = None
        
        return mock_resp
    
    async def create(self, *args, **kwargs):
        """Forward create() calls to original (not commonly used with parse)."""
        if self.mode == "realtime":
            return await self._original.create(*args, **kwargs)
        else:
            raise NotImplementedError("create() not supported in batch/cache modes, use parse()")


class WrappedChat:
    """Wrapper for chat with completions attribute."""
    
    def __init__(
        self,
        original_chat: Any,
        mode: str,
        cache_mgr: CacheManager,
        queue_mgr: BatchQueueManager,
        model: str = "",
        semaphore: Optional[asyncio.Semaphore] = None
    ):
        self._original = original_chat
        self.completions = WrappedChatCompletions(
            original_chat.completions,
            mode,
            cache_mgr,
            queue_mgr,
            model,
            semaphore
        )


class WrappedAsyncOpenAI:
    """
    Wrapper for AsyncOpenAI that intercepts API calls.
    
    Usage:
        # Instead of:
        client = AsyncOpenAI(api_key=..., base_url=...)
        
        # Use:
        client = create_wrapped_client(api_key=..., base_url=..., mode="batch_write")
    """
    
    def __init__(
        self,
        original_client: AsyncOpenAI,
        mode: str = "realtime",
        cache_dir: str = ".cache",
        task_name: Optional[str] = None,
        model: str = "",
        max_concurrency: int = 20
    ):
        """
        Initialize wrapper.
        
        Args:
            original_client: Original AsyncOpenAI client
            mode: Operation mode (realtime, batch_write, cache_first)
            cache_dir: Directory for cache storage
            task_name: Task name for batch queue organization
            model: Model name (for task info)
            max_concurrency: Maximum concurrent requests in realtime mode (default: 20)
        """
        self._original = original_client
        self.mode = mode
        self.model = model
        self.cache_mgr = CacheManager(cache_dir)
        self.queue_mgr = BatchQueueManager(cache_dir, task_name)
        
        # Create semaphore for concurrency control in realtime mode
        semaphore = asyncio.Semaphore(max_concurrency) if mode == "realtime" else None
        
        # Wrap chat
        self.chat = WrappedChat(
            original_client.chat,
            mode,
            self.cache_mgr,
            self.queue_mgr,
            model,
            semaphore
        )
        
        # Forward other attributes to original client
        self.files = original_client.files
        self.batches = original_client.batches
    
    def __getattr__(self, name: str) -> Any:
        """Forward unknown attributes to original client."""
        return getattr(self._original, name)


def create_wrapped_client(
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    mode: Optional[str] = None,
    cache_dir: str = ".cache",
    task_name: Optional[str] = None,
    model: str = "",
    strategy: str = "",
    max_concurrency: int = 20,
    **kwargs
) -> WrappedAsyncOpenAI:
    """
    Create a wrapped AsyncOpenAI client with cache and batch support.
    
    Args:
        api_key: OpenAI API key
        base_url: API base URL
        mode: Operation mode (realtime, batch_write, cache_first)
              If None, reads from MODE environment variable (default: realtime)
        cache_dir: Directory for cache storage
        task_name: Task name for batch queue (e.g., "gpt-4o_20250104_143022")
                   If None and mode is batch_write, auto-generates from model + timestamp
        model: Model name (used for task naming and info)
        strategy: Generation strategy (saved in task info)
        max_concurrency: Maximum concurrent requests in realtime mode (default: 20)
        **kwargs: Additional arguments for AsyncOpenAI
        
    Returns:
        Wrapped client that behaves like AsyncOpenAI but with cache/batch support
        
    Example:
        >>> client = create_wrapped_client(
        ...     api_key="sk-...", 
        ...     mode="batch_write",
        ...     model="gpt-4o",
        ...     strategy="full_traj_v2"
        ... )
        >>> resp = await client.chat.completions.parse(...)
    """
    if mode is None:
        mode = os.environ.get("MODE", "realtime")
    
    # Normalize mode
    if mode not in ("realtime", "batch_write", "cache_first"):
        # If mode is something else (like batch commands), default to realtime
        if mode.startswith("batch_"):
            mode = "realtime"  # Batch commands don't need wrapper
    
    # Auto-generate task name for batch_write mode
    if mode in ("batch_write", "cache_first") and task_name is None:
        # Use model name (sanitized) + timestamp
        model_safe = model.replace("/", "_").replace(" ", "_") if model else "default"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        task_name = f"{model_safe}_{timestamp}"
    
    # Create original client
    original_client = AsyncOpenAI(
        api_key=api_key,
        base_url=base_url,
        **kwargs
    )
    
    # Wrap it
    wrapped = WrappedAsyncOpenAI(
        original_client, mode, cache_dir, task_name, model, max_concurrency
    )
    
    # Create task info if in batch_write mode
    if mode == "batch_write" and task_name:
        wrapped.queue_mgr.create_task_info(model, strategy)
        print(f"[Task] Created batch task: {task_name}")
    
    return wrapped


def get_queue_stats(cache_dir: str = ".cache") -> Dict[str, Any]:
    """Get statistics about queued requests."""
    queue_mgr = BatchQueueManager(cache_dir)
    cache_mgr = CacheManager(cache_dir)
    
    queued_requests = queue_mgr.get_queued_requests()
    cache_stats = cache_mgr.get_cache_stats()
    
    return {
        "queued_requests": len(queued_requests),
        "cached_requests": cache_stats["requests"],
        "cached_responses": cache_stats["responses"],
        "cache_hit_rate": cache_stats["hit_rate"]
    }

